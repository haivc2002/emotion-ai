document.addEventListener('DOMContentLoaded', () => {
    // Tab switching logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
            
            // Stop webcam if switching to upload tab
            if (btn.dataset.tab === 'upload-tab' && stream) {
                stopWebcam();
            }
        });
    });

    // --- Webcam Logic ---
    const video = document.getElementById('videoElement');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const startBtn = document.getElementById('startWebcamBtn');
    const stopBtn = document.getElementById('stopWebcamBtn');
    const flipBtn = document.getElementById('flipCameraBtn');
    let stream = null;
    let predictionInterval = null;
    let isPredicting = false;
    let isMirrored = true; // Default to mirror
    video.classList.add('mirrored'); // Apply default mirror

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
            flipBtn.style.display = 'inline-block';
            
            // Match canvas size to video
            video.onloadedmetadata = () => {
                overlayCanvas.width = video.videoWidth;
                overlayCanvas.height = video.videoHeight;
                // Start predicting loop
                predictionInterval = setInterval(predictFrame, 500); // every 500ms
            };
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please ensure permissions are granted.");
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        if (predictionInterval) {
            clearInterval(predictionInterval);
        }
        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
        flipBtn.style.display = 'none';
        
        // Clear canvas
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        document.getElementById('resultsContent').innerHTML = '<p class="placeholder-text">Waiting for input...</p>';
    }

    async function predictFrame() {
        if (isPredicting || !video.videoWidth) return;
        isPredicting = true;

        // Create a temporary canvas to get the frame as base64
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);

        try {
            const response = await fetch('/predict_base64', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });
            const data = await response.json();
            
            if (data.error) {
                console.error(data.error);
                document.getElementById('resultsContent').innerHTML = `<p class="placeholder-text">${data.error}</p>`;
            } else {
                drawResults(data.faces, overlayCanvas, video.videoWidth, video.videoHeight, isMirrored);
                updateResultsPanel(data.faces);
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }
        
        isPredicting = false;
    }

    startBtn.addEventListener('click', startWebcam);
    stopBtn.addEventListener('click', stopWebcam);
    flipBtn.addEventListener('click', () => {
        isMirrored = !isMirrored;
        if (isMirrored) {
            video.classList.add('mirrored');
        } else {
            video.classList.remove('mirrored');
        }
    });


    // --- File Upload Logic ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.querySelector('.preview-container');
    const imagePreview = document.getElementById('imagePreview');
    const uploadCanvas = document.getElementById('uploadCanvas');
    const resetBtn = document.getElementById('resetUploadBtn');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
        dropZone.style.background = 'rgba(59, 130, 246, 0.05)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background = 'rgba(15, 23, 42, 0.3)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background = 'rgba(15, 23, 42, 0.3)';
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        previewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        fileInput.value = '';
        const ctx = uploadCanvas.getContext('2d');
        ctx.clearRect(0, 0, uploadCanvas.width, uploadCanvas.height);
        document.getElementById('resultsContent').innerHTML = '<p class="placeholder-text">Waiting for input...</p>';
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert("Please upload an image file.");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'block';
            
            imagePreview.onload = () => {
                uploadCanvas.width = imagePreview.naturalWidth;
                uploadCanvas.height = imagePreview.naturalHeight;
                predictImage(file, imagePreview.naturalWidth, imagePreview.naturalHeight);
            };
        };
        reader.readAsDataURL(file);
    }

    async function predictImage(file, width, height) {
        const formData = new FormData();
        formData.append('file', file);

        document.getElementById('resultsContent').innerHTML = '<p class="placeholder-text">Analyzing image...</p>';

        try {
            const response = await fetch('/predict_upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                document.getElementById('resultsContent').innerHTML = `<p class="placeholder-text">${data.error}</p>`;
            } else {
                drawResults(data.faces, uploadCanvas, width, height);
                updateResultsPanel(data.faces);
            }
        } catch (err) {
            console.error("Upload prediction error:", err);
            document.getElementById('resultsContent').innerHTML = '<p class="placeholder-text">Error occurred during prediction.</p>';
        }
    }


    // --- Shared Drawing Logic ---
    function drawResults(faces, canvas, originalWidth, originalHeight, mirrored = false) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!faces || faces.length === 0) return;

        // Xử lý scale nếu canvas bị thu nhỏ bởi CSS object-fit: contain
        // Việc này hơi phức tạp vì object-fit: contain scale ảnh/video nhưng giữ tỷ lệ.
        // Để đơn giản, CSS của chúng ta đặt width: 100% height: 100% object-fit: contain.
        // Canvas overlay cũng phải match exactly. Do overlay được set w/h = videoWidth/videoHeight, 
        // trình duyệt tự scale nó khớp với video.
        
        ctx.lineWidth = 3;
        ctx.font = "bold 24px Inter, sans-serif";

        faces.forEach(face => {
            let [x, y, w, h] = face.box;
            
            if (mirrored) {
                x = originalWidth - x - w;
            }
            
            // Draw box
            ctx.strokeStyle = "#3b82f6";
            ctx.strokeRect(x, y, w, h);
            
            // Draw text background
            const text = `${face.emotion} (${(face.confidence * 100).toFixed(1)}%)`;
            ctx.fillStyle = "#3b82f6";
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(x, y - 30, textWidth + 10, 30);
            
            // Draw text
            ctx.fillStyle = "#ffffff";
            ctx.fillText(text, x + 5, y - 8);
        });
    }

    function updateResultsPanel(faces) {
        const resultsContent = document.getElementById('resultsContent');
        if (!faces || faces.length === 0) {
            resultsContent.innerHTML = '<p class="placeholder-text">No faces detected.</p>';
            return;
        }

        let html = '';
        faces.forEach((face, idx) => {
            html += `<span class="emotion-badge">Face ${idx + 1}: ${face.emotion} (${(face.confidence * 100).toFixed(1)}%)</span>`;
        });
        resultsContent.innerHTML = html;
    }
});
