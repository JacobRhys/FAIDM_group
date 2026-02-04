import os
import pathlib

def create_structure():
    # Define the base directory (django_app)
    base_dir = pathlib.Path("django_app")

    # Define the folder structure required by the plan
    directories = [
        base_dir / "core",
        base_dir / "student_risk",
        base_dir / "student_risk" / "migrations",
        base_dir / "student_risk" / "static" / "css",      # For Tailwind styles [cite: 10]
        base_dir / "student_risk" / "static" / "js",       # For animations/simulator [cite: 11]
        base_dir / "student_risk" / "templates",           # For HTML templates [cite: 13]
    ]

    # Define the files to create
    files = [
        # Root Files
        base_dir / "manage.py",                            # Standard Django entry point [cite: 6]
        
        # Core Project Settings
        base_dir / "core" / "__init__.py",
        base_dir / "core" / "settings.py",                 # Project settings [cite: 6]
        base_dir / "core" / "urls.py",
        base_dir / "core" / "wsgi.py",
        base_dir / "core" / "asgi.py",

        # Student Risk App Logic
        base_dir / "student_risk" / "__init__.py",
        base_dir / "student_risk" / "admin.py",
        base_dir / "student_risk" / "apps.py",
        base_dir / "student_risk" / "models.py",
        base_dir / "student_risk" / "tests.py",
        base_dir / "student_risk" / "urls.py",
        base_dir / "student_risk" / "views.py",            # For index, dashboard, predict_api 
        base_dir / "student_risk" / "services.py",         # For ModelWrapper class 

        # Frontend Assets
        base_dir / "student_risk" / "static" / "css" / "styles.css",   # Tailwind input [cite: 10]
        base_dir / "student_risk" / "static" / "js" / "animations.js", # Scroll reveals [cite: 11]
        base_dir / "student_risk" / "static" / "js" / "simulator.js",  # AJAX logic [cite: 12]

        # Templates
        base_dir / "student_risk" / "templates" / "base.html",         # Layout with navbar/footer [cite: 13]
        base_dir / "student_risk" / "templates" / "index.html",        # Hero section [cite: 14]
        base_dir / "student_risk" / "templates" / "dashboard.html",    # Bento Grid [cite: 14]
    ]

    print(f"Creating project structure in '{base_dir.resolve()}'...")

    # Create directories
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except OSError as e:
            print(f"❌ Error creating directory {directory}: {e}")

    # Create empty files
    for file_path in files:
        try:
            if not file_path.exists():
                file_path.touch()
                print(f"✅ Created file: {file_path}")
            else:
                print(f"⚠️  File already exists: {file_path}")
        except OSError as e:
            print(f"❌ Error creating file {file_path}: {e}")

    print("\nStructure created successfully.")

if __name__ == "__main__":
    create_structure()