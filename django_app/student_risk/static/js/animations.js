document.addEventListener('DOMContentLoaded', () => {
    // Scroll Reveal Logic using IntersectionObserver
    const observerOptions = {
        threshold: 0.1,
        rootMargin: "0px 0px -100px 0px" // Start animating slightly before they come fully into view
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // We keep observing if we want them to re-animate on scroll up/down
                // or unobserve if one-time: observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach((el, index) => {
        // Automatically add slight delay based on position for sequential feel
        if (!el.style.transitionDelay) {
            el.style.transitionDelay = `${(index % 3) * 0.1}s`;
        }
        observer.observe(el);
    });

    // Subtle Parallax & Floating effects for background elements
    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;

        // Move the mesh background slightly to create depth
        const mesh = document.querySelector('.mesh-gradient');
        if (mesh) {
            mesh.style.transform = `scale(1.1) rotate(${5 + scrolled * 0.01}deg) translateY(${scrolled * 0.1}px)`;
        }
    });
});
