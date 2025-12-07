
document.addEventListener('DOMContentLoaded', function () {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const loadingOverlay = document.getElementById('loading-overlay');
    const form = document.getElementById('upload-form');

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    // Handle drop
    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                // Show preview
                const reader = new FileReader();
                reader.onload = function (e) {
                    previewImage.src = e.target.result;
                    previewContainer.style.display = 'block';

                    // Auto submit after a short delay to show preview
                    loadingOverlay.classList.add('active');

                    // Assign file to input if it came from drop
                    if (fileInput.files.length === 0 || fileInput.files[0] !== file) {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                    }

                    form.submit();
                }
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    // Animate confidence bar if present
    const confidenceBar = document.querySelector('.confidence-fill');
    if (confidenceBar) {
        const width = confidenceBar.getAttribute('data-width');
        setTimeout(() => {
            confidenceBar.style.width = width;
        }, 300);
    }
});
