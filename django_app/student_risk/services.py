import joblib
import pandas as pd
import numpy as np
import os
from django.conf import settings
from sklearn.metrics import accuracy_score, f1_score

class ModelWrapper:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelWrapper, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance
    
    def load_model(self):
        model_path = os.path.join(settings.BASE_DIR.parent, 'student_success_model.joblib')
        print(f"Loading model from: {model_path}")
        
        try:
            self.artifacts = joblib.load(model_path)
            self.model = self.artifacts['model']
            self.imputer = self.artifacts['imputer']
            self.calibrator = self.artifacts.get('calibrator')
            self.threshold = self.artifacts.get('best_threshold', 0.5)
            self.scale_pos = self.artifacts.get('scale_pos')
            print("Model loaded successfully!")
            
            self.student_df = None
            self.feature_names = None
            try:
                master_path = os.path.join(settings.BASE_DIR.parent, 'students_cleaned_with_id.csv')
                if os.path.exists(master_path):
                    master_df = pd.read_csv(master_path, nrows=1, skipinitialspace=True) # Added skipinitialspace
                    master_df.columns = [c.strip() for c in master_df.columns]
                    drop_cols = ['final_result', 'date_unregistration']
                    self.feature_names = [c for c in master_df.columns if c not in drop_cols]
            except Exception as fe:
                print(f"Warning: Could not detect feature names: {fe}")
                
        except Exception as e:
            print(f"CRITICAL ERROR loading model/data: {e}")
            self.model = None
            self.student_df = None

    def _align_features(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([df])

        median_profile = {}
        if self.student_df is not None:
            exclude_cols = ['final_result', 'date_unregistration']
            numeric_df = self.student_df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in self.student_df.columns], errors='ignore')
            median_profile = numeric_df.median().to_dict()

        for col, val in median_profile.items():
            if col not in df.columns:
                df[col] = val

        expected_cols = self.feature_names
        if expected_cols is None and hasattr(self.model, 'feature_names_in_'):
            expected_cols = self.model.feature_names_in_

        if expected_cols is not None:
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_cols]
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _predict_internal(self, df):
        if not self.model: return None
        X = self._align_features(df)
        probs = self.model.predict_proba(X)[:, 1]
        if self.calibrator:
            probs = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        return probs

    def process_bulk_predictions(self, csv_file):
        if not self.model: return {'error': 'Model not loaded'}

        try:
            df = pd.read_csv(csv_file, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df = df.replace(r'^\s*$', np.nan, regex=True)
            
            if df.empty: return {'error': 'Uploaded CSV is empty'}

            self.student_df = df
            probs = self._predict_internal(df)
            if probs is None: return {'error': 'Model not loaded'}

            # --- Metrics ---
            at_risk_count = np.sum(probs >= self.threshold)
            at_risk_rate = (at_risk_count / len(df)) * 100
            
            metrics = {
                'total_students': len(df),
                'at_risk_rate': round(at_risk_rate, 1),
                'accuracy': None, 'f1_score': None
            }

            if 'final_result' in df.columns:
                # Robust target extraction
                numeric_series = pd.to_numeric(df['final_result'], errors='coerce')
                if numeric_series.notna().sum() > 0.8 * len(df):
                     y_true = (numeric_series < 0).astype(int)
                     valid_mask = numeric_series.notna()
                     y_true = y_true[valid_mask]
                else:
                    y_true = df['final_result'].map({'Pass': 0, 'Distinction': 0, 'Fail': 1, 'Withdrawn': 1}).dropna()

                temp_df = pd.DataFrame({'prob': probs}).reset_index(drop=True)
                temp_df['y_true'] = y_true.values if len(y_true) == len(probs) else y_true.reset_index(drop=True)
                temp_eval = temp_df.dropna(subset=['y_true'])
                
                if not temp_eval.empty:
                    y_pred_subset = (temp_eval['prob'].values >= self.threshold).astype(int)
                    y_true_subset = temp_eval['y_true'].values.astype(int)
                    metrics['accuracy'] = round(accuracy_score(y_true_subset, y_pred_subset) * 100, 1)
                    metrics['f1_score'] = round(f1_score(y_true_subset, y_pred_subset) * 100, 1)

            # --- DIVERSIFIED SELECTION LOGIC ---
            result_df = df.copy()
            result_df['risk_score'] = probs
            
            # Detect Columns
            interaction_col = 'total_click_sum' if 'total_click_sum' in df.columns else 'total_click_count'
            perf_col = 'final_result' if 'final_result' in df.columns else 'studied_credits'

            def classify_student(row):
                if not interaction_col or not perf_col: return "Unclassified"
                try:
                    interact_val = float(row[interaction_col])
                    perf_val = float(row[perf_col])
                    
                    # Thresholds: 0 is Mean (Z-score) or Baseline
                    high_interact = interact_val > -0.5 # More active than the very bottom
                    good_perform = perf_val > -0.5      # Better performance than the very bottom

                    if high_interact and not good_perform: return 'Struggling'
                    elif not high_interact and good_perform: return 'Ghost Achiever'
                    elif high_interact and good_perform: return 'Star Student'
                    else: return 'Disengaged'
                except: return "Unclassified"

            result_df['category'] = result_df.apply(classify_student, axis=1)

            # STRATEGY: Return Top 250 highest-risk students to support dashboard toggles
            # The classification (Struggling/Disengaged/etc.) provides valuable behavioral context
            critical_df = result_df.sort_values('risk_score', ascending=False).head(250)

            # Format Function
            def format_row(row):
                return {
                    'id_student': int(row['id_student']) if 'id_student' in row and pd.notna(row['id_student']) else 'N/A',
                    'risk_score': round(float(row['risk_score']), 4),
                    'studied_credits': int(row['studied_credits']) if 'studied_credits' in row and pd.notna(row['studied_credits']) else 0,
                    'disability': 'Yes' if 'disability' in row and pd.notna(row['disability']) and row['disability'] == 1 else 'No',
                    'category': row.get('category', 'Unclassified')
                }

            critical_students = [format_row(row) for _, row in critical_df.iterrows()]
            all_students = [format_row(row) for _, row in result_df.iterrows()]

            response = metrics
            response['critical_students'] = critical_students
            response['all_students'] = all_students
            
            self.student_df = df
            return response

        except Exception as e:
            print(f"Error processing bulk predictions: {e}")
            return {'error': f"Failed to process CSV: {str(e)}"}
            
    # Keep other methods (predict, get_student_risk) as they were...
    def predict(self, feature_dict):
        try:
            probs = self._predict_internal(pd.DataFrame([feature_dict]))
            if probs is None: return {'error': 'Model not loaded'}
            risk_score = float(probs[0])
            return {'risk_score': risk_score, 'is_at_risk': risk_score >= self.threshold, 'threshold': self.threshold}
        except Exception as e: return {'error': str(e)}

    def get_student_risk(self, student_id):
        if self.student_df is None: return {'error': 'No dataset available.'}
        
        # Ensure student_id is int for matching
        try:
            student_id = int(student_id)
        except: pass
            
        student_record = self.student_df[self.student_df['id_student'] == student_id]
        if student_record.empty: return {'error': f'Student ID {student_id} not found'}
        
        row = student_record.iloc[0]
        try:
            X_input = student_record.drop(['id_student', 'final_result'], axis=1, errors='ignore')
            probs = self._predict_internal(X_input)
            risk_score = float(probs[0])
            
            # Recalculate Category for this specific student if not already present
            interaction_col = 'total_click_sum' if 'total_click_sum' in self.student_df.columns else 'total_click_count'
            perf_col = 'final_result' if 'final_result' in self.student_df.columns else 'studied_credits'
            
            # Simple version of category logic for single lookup
            category = "Unknown"
            try:
                interact_val = float(row[interaction_col])
                perf_val = float(row[perf_col])
                high_interact = interact_val > -0.5
                good_perform = perf_val > -0.5
                if high_interact and not good_perform: category = 'Struggling'
                elif not high_interact and good_perform: category = 'Ghost Achiever'
                elif high_interact and good_perform: category = 'Star Student'
                else: category = 'Disengaged'
            except: pass

            # Calculate Averages for Comparison
            avg_clicks = self.student_df[interaction_col].mean()
            avg_credits = self.student_df['studied_credits'].mean()

            return {
                'id_student': int(student_id),
                'risk_score': round(risk_score * 100, 1),
                'risk_level': 'High' if risk_score >= 0.5 else 'Low',
                'credits': int(row['studied_credits']) if 'studied_credits' in row else 0,
                'clicks': int(row[interaction_col]) if interaction_col in row else 0,
                'imd_band': row['imd_band'] if 'imd_band' in row else 'N/A',
                'disability': 'Yes' if ('disability' in row and row['disability'] == 1) else 'No',
                'category': category,
                'gender': row.get('gender', 'N/A'),
                'region': row.get('region', 'N/A'),
                'education': row.get('highest_education', 'N/A'),
                'age': row.get('age_band', 'N/A'),
                'avg_clicks': round(avg_clicks, 1),
                'avg_credits': round(avg_credits, 1)
            }
        except Exception as e: return {'error': str(e)}