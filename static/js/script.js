document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const filenameDisplay = document.getElementById('filename');
    const removeFileBtn = document.getElementById('remove-file');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('result-section');
    const resultText = document.getElementById('result-text');
    const statusIcon = document.getElementById('status-icon');
    const confidenceValue = document.getElementById('confidence-value');
    const progressFill = document.getElementById('progress-fill');
    const browseBtn = document.querySelector('.browse-btn');

    let currentFile = null;

    // Drag & Drop Handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Click to Browse
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeFileBtn) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Remove File
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        analyzeBtn.disabled = true;
        resultSection.classList.remove('active');
        resetUI();
    });

    function handleFile(file) {
        if (file.type.startsWith('video/')) {
            currentFile = file;
            filenameDisplay.textContent = file.name;
            fileInfo.style.display = 'flex';
            analyzeBtn.disabled = false;
            resultSection.classList.remove('active');
        } else {
            alert('Please upload a valid video file.');
        }
    }

    function resetUI() {
        dropZone.querySelector('h3').style.display = 'block';
        dropZone.querySelector('p').style.display = 'block';
        dropZone.querySelector('.icon-container').style.display = 'block';
    }

    // Analyze Button Click
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('span').textContent = 'Processing...';
        loader.style.display = 'inline-block';
        resultSection.classList.remove('active');

        const formData = new FormData();
        formData.append('video', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data.label, data.confidence);
            } else {
                alert('Error: ' + (data.error || 'Something went wrong'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server.');
        } finally {
            // Reset Button State
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('span').textContent = 'Analyze Video';
            loader.style.display = 'none';
        }
    });

    function showResult(label, confidence) {
        resultSection.classList.add('active');

        // Format confidence to percentage
        const percentage = Math.round(confidence * 100);

        resultText.textContent = `${label} VIDEO DETECTED`;
        confidenceValue.textContent = `${percentage}%`;
        progressFill.style.width = `${percentage}%`;

        // Update Styles based on result
        if (label === 'FAKE') {
            statusIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            statusIcon.className = 'status-icon fake';
            progressFill.style.backgroundColor = '#ef4444';
            resultText.style.color = '#ef4444';
        } else {
            statusIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            statusIcon.className = 'status-icon real';
            progressFill.style.backgroundColor = '#10b981';
            resultText.style.color = '#10b981';
        }
    }
});
