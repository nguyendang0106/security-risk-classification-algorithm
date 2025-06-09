document.addEventListener('DOMContentLoaded', () => {
    // === EXISTING VARIABLES ===
    const pageMainSelection = document.getElementById('pageMainSelection');
    const inputAreaContainer = document.getElementById('inputAreaContainer');
    const directInputSection = document.getElementById('directInputSection');
    const fileUploadSection = document.getElementById('fileUploadSection');
    const btnShowDirectInputPage = document.getElementById('btnShowDirectInputPage');
    const btnShowFileUploadPage = document.getElementById('btnShowFileUploadPage');
    const backToSelectionButton = document.getElementById('backToSelectionButton');
    const featureInput = document.getElementById('featureInput');
    const fileInput = document.getElementById('fileInput');
    const predictButton = document.getElementById('predictButton');
    const predictionList = document.getElementById('predictionList');
    const singlePredictionResult = document.getElementById('singlePredictionResult');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const apiUrlInput = document.getElementById('apiUrl');
    const expectedFeaturesCountSpan = document.getElementById('expectedFeaturesCount');
    const themeToggleButton = document.getElementById('themeToggleButton');
    const carouselTextElement = document.getElementById('carouselText');
    const body = document.body;
    // New elements for System Monitoring
    const pageSystemMonitoring = document.getElementById('pageSystemMonitoring');
    const btnShowSystemMonitoringPage = document.getElementById('btnShowSystemMonitoringPage');
    const backToSelectionFromMonitoringButton = document.getElementById('backToSelectionFromMonitoringButton');
    
    const monitoringFileInput = document.getElementById('monitoringFileInput');
    const uploadMonitoringCsvButton = document.getElementById('uploadMonitoringCsvButton');
    const monitoringFileStatus = document.getElementById('monitoringFileStatus');
    
    const monitoringControlsSection = document.getElementById('monitoringControlsSection');
    const startMonitoringButton = document.getElementById('startMonitoringButton');
    const stopMonitoringButton = document.getElementById('stopMonitoringButton');
    const monitoringStatus = document.getElementById('monitoringStatus');
    const monitoringApiUrlInput = document.getElementById('monitoringApiUrl');
    
    const monitoringResultsSection = document.getElementById('monitoringResultsSection');
    const latestMonitoringPrediction = document.getElementById('latestMonitoringPrediction');
    const latestMonitoringInput = document.getElementById('latestMonitoringInput');
    const allMonitoringPredictionsList = document.getElementById('allMonitoringPredictionsList');
    const monitoringProgressCount = document.getElementById('monitoringProgressCount');
    
    const monitoringErrorDisplay = document.getElementById('monitoringErrorDisplay');
    const monitoringErrorMessageText = document.getElementById('monitoringErrorMessageText');
    // New elements for System Monitoring


    const N_EXPECTED_FEATURES = 67;
    if(expectedFeaturesCountSpan) expectedFeaturesCountSpan.textContent = N_EXPECTED_FEATURES;
    let activeInputType = null;
    let currentPredictionOperationId = null;

    // === ENHANCED CURSOR EFFECTS ===
    const cursorTrail = document.getElementById('cursorTrail');
    const cursorGlow = document.getElementById('cursorGlow');
    const cursorField = document.getElementById('cursorField');
    const cursorRipples = document.getElementById('cursorRipples');
    
    let mouseX = 0;
    let mouseY = 0;
    let trailX = 0;
    let trailY = 0;
    let glowX = 0;
    let glowY = 0;
    let fieldX = 0;
    let fieldY = 0;
    let isMoving = false;
    let moveTimeout;
    // New elements for System Monitoring
    let monitoringIntervalId = null;
    let isMonitoringGloballyActive = false; // To control the interval globally
    // New elements for System Monitoring

    // Enhanced mouse tracking with movement detection
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        isMoving = true;
        clearTimeout(moveTimeout);
        moveTimeout = setTimeout(() => {
            isMoving = false;
        }, 100);
        
        // Show magnetic field when moving
        if (cursorField) {
            cursorField.style.opacity = isMoving ? '0.6' : '0';
        }
    });

    // Enhanced ripple effect on click
    document.addEventListener('mousedown', (e) => {
        createRipple(e.clientX, e.clientY);
        if (cursorTrail) {
            cursorTrail.classList.add('click');
            setTimeout(() => {
                cursorTrail.classList.remove('click');
            }, 300);
        }
    });

    // --- Navigation for Monitoring Page ---
    function showSystemMonitoringPageUI() {
        currentPredictionOperationId = null; // Cancel any ongoing single/batch predictions
        switchPage(pageMainSelection, pageSystemMonitoring, 'forward'); // Ensure switchPage is defined
        
        // Reset monitoring UI elements to initial state
        if(monitoringFileStatus) monitoringFileStatus.textContent = 'Chưa có tệp nào được tải lên.';
        if(monitoringControlsSection) monitoringControlsSection.style.display = 'none';
        if(monitoringResultsSection) monitoringResultsSection.style.display = 'none';
        if(allMonitoringPredictionsList) allMonitoringPredictionsList.innerHTML = '';
        if(latestMonitoringPrediction) latestMonitoringPrediction.textContent = '---';
        if(latestMonitoringInput) latestMonitoringInput.textContent = 'Đầu vào: ---';
        if(monitoringProgressCount) monitoringProgressCount.textContent = '0/0';
        if(monitoringStatus) monitoringStatus.textContent = 'Trạng thái: Chưa hoạt động';
        if(monitoringErrorDisplay) monitoringErrorDisplay.style.display = 'none';
        if(monitoringFileInput) monitoringFileInput.value = '';
        if(startMonitoringButton) startMonitoringButton.disabled = true;
        if(stopMonitoringButton) stopMonitoringButton.style.display = 'none';
        
        stopMonitoringProcess(); // Ensure any previous interval is cleared
    }

    if(btnShowSystemMonitoringPage) {
        btnShowSystemMonitoringPage.addEventListener('click', showSystemMonitoringPageUI);
    }
    if(backToSelectionFromMonitoringButton) {
        backToSelectionFromMonitoringButton.addEventListener('click', () => {
            stopMonitoringProcess(true); // true to also call backend to stop
            switchPage(pageSystemMonitoring, pageMainSelection, 'backward');
        });
    }

    // --- System Monitoring Logic ---
    if(uploadMonitoringCsvButton) {
        uploadMonitoringCsvButton.addEventListener('click', async () => {
            if (!monitoringFileInput.files.length) {
                showMonitoringErrorMsg("Vui lòng chọn một tệp CSV.");
                return;
            }
            const file = monitoringFileInput.files[0];
            const formData = new FormData();
            formData.append("file", file);

            monitoringFileStatus.textContent = "Đang tải lên và chuẩn bị tệp...";
            monitoringFileStatus.style.color = "var(--text-color-secondary)";
            monitoringErrorDisplay.style.display = 'none';
            startMonitoringButton.disabled = true;

            try {
                // Use the API URL from the main page if needed, or hardcode for this specific upload
                const uploadApiUrl = document.getElementById('apiUrl').value.replace('/predict', '/upload_monitoring_csv') || 'http://127.0.0.1:8888/upload_monitoring_csv';
                const response = await fetch(uploadApiUrl, {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || `Lỗi máy chủ: ${response.status}`);
                }
                monitoringFileStatus.textContent = `Tệp '${data.fileName}' (${data.rowCount} dòng) đã tải lên. Sẵn sàng giám sát.`;
                monitoringFileStatus.style.color = "var(--accent-color-secondary)";
                monitoringControlsSection.style.display = 'block';
                monitoringResultsSection.style.display = 'block';
                allMonitoringPredictionsList.innerHTML = '';
                latestMonitoringPrediction.textContent = '---';
                latestMonitoringInput.textContent = 'Đầu vào: ---';
                monitoringProgressCount.textContent = `0/${data.rowCount}`;
                startMonitoringButton.disabled = false;
                stopMonitoringButton.style.display = 'none';

            } catch (error) {
                showMonitoringErrorMsg(`Lỗi tải tệp: ${error.message}`);
                monitoringFileStatus.textContent = "Tải tệp thất bại.";
                monitoringFileStatus.style.color = "var(--error-color)";
            }
        });
    }

    if(startMonitoringButton) {
        startMonitoringButton.addEventListener('click', async () => {
            monitoringStatus.textContent = 'Trạng thái: Đang khởi động...';
            monitoringErrorDisplay.style.display = 'none';
            allMonitoringPredictionsList.innerHTML = ''; 
            latestMonitoringPrediction.textContent = '---';
            latestMonitoringInput.textContent = 'Đầu vào: ---';

            try {
                const startApiUrl = document.getElementById('apiUrl').value.replace('/predict', '/start_monitoring') || 'http://127.0.0.1:8888/start_monitoring';
                const response = await fetch(startApiUrl, { method: 'POST' });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || "Không thể bắt đầu giám sát từ API.");

                isMonitoringGloballyActive = true;
                startMonitoringButton.style.display = 'none';
                stopMonitoringButton.style.display = 'inline-block';
                monitoringStatus.textContent = 'Trạng thái: Đang hoạt động...';
                
                fetchMonitoringUpdateData(); // Initial fetch
                if (monitoringIntervalId) clearInterval(monitoringIntervalId);
                monitoringIntervalId = setInterval(fetchMonitoringUpdateData, 1000);
            } catch (error) {
                showMonitoringErrorMsg(`Lỗi bắt đầu giám sát: ${error.message}`);
                monitoringStatus.textContent = 'Trạng thái: Lỗi khởi động.';
            }
        });
    }

    if(stopMonitoringButton) {
        stopMonitoringButton.addEventListener('click', () => {
            stopMonitoringProcess(true); // true to call backend stop
        });
    }

    async function stopMonitoringProcess(callApiToStop = false) {
        isMonitoringGloballyActive = false;
        if (monitoringIntervalId) {
            clearInterval(monitoringIntervalId);
            monitoringIntervalId = null;
        }
        if(startMonitoringButton) {
            startMonitoringButton.disabled = false; // Re-enable if a file is loaded
            startMonitoringButton.style.display = 'inline-block';
        }
        if(stopMonitoringButton) stopMonitoringButton.style.display = 'none';
        if(monitoringStatus) {
            // Keep last status if it was 'finished' or 'error'
            if (monitoringStatus.textContent.toLowerCase().includes('hoàn thành') || monitoringStatus.textContent.toLowerCase().includes('lỗi')) {
                // do nothing
            } else {
                monitoringStatus.textContent = 'Trạng thái: Đã dừng.';
            }
        }

        if (callApiToStop) {
            try {
                const stopApiUrl = document.getElementById('apiUrl').value.replace('/predict', '/stop_monitoring') || 'http://127.0.0.1:8888/stop_monitoring';
                await fetch(stopApiUrl, { method: 'POST' });
            } catch (error) {
                console.warn("Lỗi khi gọi API dừng giám sát (có thể bỏ qua):", error);
            }
        }
    }

    async function fetchMonitoringUpdateData() {
        if (!isMonitoringGloballyActive) return;

        const getUpdateApiUrl = monitoringApiUrlInput.value.trim();
        if (!getUpdateApiUrl) {
            showMonitoringErrorMsg("URL API giám sát không được để trống.");
            stopMonitoringProcess();
            return;
        }

        try {
            const response = await fetch(getUpdateApiUrl);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `Lỗi API giám sát: ${response.status}`);
            }
            
            // Update latest prediction display
            if (data.latest_prediction) {
                const pred = data.latest_prediction;
                latestMonitoringPrediction.textContent = pred.prediction;
                latestMonitoringPrediction.style.color = pred.is_error ? 'var(--error-color)' : getPredictionColor(pred.prediction);
                
                let inputRowDisplay = `Dòng ${pred.line_index + 1}: ${pred.input_row_str}`;
                const paddedCount = pred.padded_count || 0;
                const truncatedCount = pred.truncated_count || 0;

                if (paddedCount > 0) {
                    inputRowDisplay += ` (đã thêm ${paddedCount} đặc trưng 0)`;
                }
                if (truncatedCount > 0) {
                    inputRowDisplay += ` (đã bỏ ${truncatedCount} đặc trưng thừa)`;
                }
                latestMonitoringInput.textContent = inputRowDisplay;

            } else if (data.status !== "finished" && data.status !== "idle") {
                latestMonitoringPrediction.textContent = '---';
                latestMonitoringInput.textContent = 'Đầu vào: ---';
            }

            
            // Update all predictions list
            if (data.latest_prediction && allMonitoringPredictionsList) {
                 const pred = data.latest_prediction;
                 const listItem = document.createElement('li');
                 
                 let listItemText = `Dòng ${pred.line_index + 1}: "${pred.input_row_str}" -> ${pred.prediction}`;
                 const paddedCount = pred.padded_count || 0;
                 const truncatedCount = pred.truncated_count || 0;

                 if (paddedCount > 0) {
                    listItemText += ` (đã thêm ${paddedCount} đặc trưng 0)`;
                 }
                 if (truncatedCount > 0) {
                    listItemText += ` (đã bỏ ${truncatedCount} đặc trưng thừa)`;
                 }
                 listItem.textContent = listItemText;
                 listItem.style.color = pred.is_error ? 'var(--error-color)' : getPredictionColor(pred.prediction);
                 
                 if (allMonitoringPredictionsList.firstChild) {
                     allMonitoringPredictionsList.insertBefore(listItem, allMonitoringPredictionsList.firstChild);
                 } else {
                     allMonitoringPredictionsList.appendChild(listItem);
                 }
            }

            // Update progress count
            if(monitoringProgressCount) monitoringProgressCount.textContent = `${data.processed_lines}/${data.total_lines}`;

            if (data.status === "finished") {
                monitoringStatus.textContent = `Trạng thái: Hoàn thành. ${data.message || ''}`;
                stopMonitoringProcess(); // Stop polling, backend already knows
            } else if (data.status === "idle") {
                monitoringStatus.textContent = 'Trạng thái: Đã dừng (theo API).';
                stopMonitoringProcess(); // Backend is idle, so stop polling
            } else { // processing
                 monitoringStatus.textContent = 'Trạng thái: Đang hoạt động...';
            }

        } catch (error) {
            showMonitoringErrorMsg(`Lỗi khi lấy cập nhật giám sát: ${error.message}`);
            monitoringStatus.textContent = 'Trạng thái: Lỗi kết nối.';
            stopMonitoringProcess(); // Stop polling on error
        }
    }

    function getPredictionColor(predictionLabel) {
        const label = predictionLabel ? predictionLabel.trim().toLowerCase() : "";
        if (label === 'benign') return 'var(--accent-color-secondary)'; // Green - Giữ nguyên
        if (label === 'unknown') return 'var(--error-color)'; // Đổi thành Đỏ cho Unknown
        if (label.toLowerCase().startsWith('error:')) return 'var(--error-color)'; // Red cho lỗi tường minh - Giữ nguyên
        // Các trường hợp còn lại (tấn công cụ thể) sẽ là màu vàng
        return 'var(--accent-color-warning)'; // Đổi thành Vàng/Cảnh báo cho các loại tấn công cụ thể
    }

    function showMonitoringErrorMsg(message) {
        if(monitoringErrorMessageText) monitoringErrorMessageText.textContent = message;
        if(monitoringErrorDisplay) monitoringErrorDisplay.style.display = 'block';
    }

    // Create ripple effect
    function createRipple(x, y) {
        if (!cursorRipples) return;
        
        const ripple = document.createElement('div');
        ripple.className = 'ripple';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        
        cursorRipples.appendChild(ripple);
        
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 800);
    }

    // Smooth cursor following animation with multiple layers
    function updateCursor() {
        const trailSpeed = 0.2;
        const glowSpeed = 0.1;
        const fieldSpeed = 0.05;

        // Update positions with different speeds for layered effect
        trailX += (mouseX - trailX) * trailSpeed;
        trailY += (mouseY - trailY) * trailSpeed;
        
        glowX += (mouseX - glowX) * glowSpeed;
        glowY += (mouseY - glowY) * glowSpeed;
        
        fieldX += (mouseX - fieldX) * fieldSpeed;
        fieldY += (mouseY - fieldY) * fieldSpeed;

        // Apply positions
        if (cursorTrail) {
            cursorTrail.style.left = trailX + 'px';
            cursorTrail.style.top = trailY + 'px';
        }

        if (cursorGlow) {
            cursorGlow.style.left = glowX + 'px';
            cursorGlow.style.top = glowY + 'px';
        }

        if (cursorField) {
            cursorField.style.left = fieldX + 'px';
            cursorField.style.top = fieldY + 'px';
        }

        requestAnimationFrame(updateCursor);
    }
    updateCursor();

    // Enhanced interactive elements hover effects
    const interactiveElements = document.querySelectorAll('button, input, textarea, a, [role="button"], li');
    
    interactiveElements.forEach(element => {
        element.addEventListener('mouseenter', () => {
            if (cursorTrail) cursorTrail.classList.add('hover');
            if (cursorGlow) cursorGlow.classList.add('hover');
            if (cursorField) cursorField.classList.add('hover');
        });

        element.addEventListener('mouseleave', () => {
            if (cursorTrail) cursorTrail.classList.remove('hover');
            if (cursorGlow) cursorGlow.classList.remove('hover');
            if (cursorField) cursorField.classList.remove('hover');
        });
    });

    // Hide cursor effects when mouse leaves window
    document.addEventListener('mouseleave', () => {
        if (cursorTrail) cursorTrail.style.opacity = '0';
        if (cursorGlow) cursorGlow.style.opacity = '0';
        if (cursorField) cursorField.style.opacity = '0';
    });

    document.addEventListener('mouseenter', () => {
        if (cursorTrail) cursorTrail.style.opacity = '1';
        if (cursorGlow) cursorGlow.style.opacity = '1';
    });

    // === ENHANCED PARTICLE SYSTEM ===
    const canvas = document.getElementById('particleCanvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    let particles = [];
    let animationId;

    function initCanvas() {
        if (!canvas || !ctx) return;
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    class EnhancedParticle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 3 + 1;
            this.speedX = (Math.random() - 0.5) * 0.8;
            this.speedY = (Math.random() - 0.5) * 0.8;
            this.opacity = Math.random() * 0.6 + 0.2;
            this.hue = Math.random() * 60 + 200; // Blue-purple range
            this.pulseSpeed = Math.random() * 0.02 + 0.01;
            this.pulsePhase = Math.random() * Math.PI * 2;
            this.magneticForce = 0;
            
            this.updateColor();
        }

        updateColor() {
            const isDark = body.classList.contains('dark-theme');
            this.color = isDark ? 
                `hsla(${this.hue}, 70%, 70%, ${this.opacity})` : 
                `hsla(${this.hue}, 60%, 50%, ${this.opacity})`;
        }

        update() {
            // Normal movement
            this.x += this.speedX;
            this.y += this.speedY;

            // Pulsing effect
            this.pulsePhase += this.pulseSpeed;
            const pulseFactor = Math.sin(this.pulsePhase) * 0.5 + 1;
            this.currentSize = this.size * pulseFactor;

            // Enhanced mouse interaction with magnetic effect
            const dx = mouseX - this.x;
            const dy = mouseY - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 150) {
                const force = (150 - distance) / 150;
                this.magneticForce = force;
                
                // Repulsion effect
                this.x -= dx * force * 0.02;
                this.y -= dy * force * 0.02;
                
                // Increase opacity when near cursor
                this.opacity = Math.min(0.8, this.opacity + force * 0.3);
            } else {
                this.magneticForce *= 0.95; // Fade magnetic effect
                this.opacity *= 0.98; // Fade opacity
                if (this.opacity < 0.2) this.opacity = 0.2;
            }

            // Wrap around edges
            if (this.x < 0) this.x = canvas.width;
            if (this.x > canvas.width) this.x = 0;
            if (this.y < 0) this.y = canvas.height;
            if (this.y > canvas.height) this.y = 0;
            
            this.updateColor();
        }

        draw() {
            // Main particle
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.currentSize, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
            
            // Glow effect when magnetized
            if (this.magneticForce > 0.1) {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.currentSize * 3, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${this.hue}, 70%, 70%, ${this.magneticForce * 0.1})`;
                ctx.fill();
            }
        }
    }

    function createParticles() {
        particles = [];
        const particleCount = Math.min(60, Math.floor(window.innerWidth / 15));
        
        for (let i = 0; i < particleCount; i++) {
            particles.push(new EnhancedParticle());
        }
    }

    function animateParticles() {
        if (!ctx || !canvas) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Update and draw particles
        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });

        // Enhanced connection lines
        particles.forEach((particle, i) => {
            particles.slice(i + 1).forEach(otherParticle => {
                const dx = particle.x - otherParticle.x;
                const dy = particle.y - otherParticle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 120) {
                    const opacity = (120 - distance) / 120 * 0.3;
                    const gradient = ctx.createLinearGradient(
                        particle.x, particle.y,
                        otherParticle.x, otherParticle.y
                    );
                    
                    const isDark = body.classList.contains('dark-theme');
                    const color1 = isDark ? `rgba(96, 165, 250, ${opacity})` : `rgba(102, 126, 234, ${opacity})`;
                    const color2 = isDark ? `rgba(139, 92, 246, ${opacity})` : `rgba(118, 75, 162, ${opacity})`;
                    
                    gradient.addColorStop(0, color1);
                    gradient.addColorStop(1, color2);
                    
                    ctx.beginPath();
                    ctx.moveTo(particle.x, particle.y);
                    ctx.lineTo(otherParticle.x, otherParticle.y);
                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            });
        });

        animationId = requestAnimationFrame(animateParticles);
    }

    function initParticleSystem() {
        initCanvas();
        createParticles();
        animateParticles();
    }

    // Handle resize
    window.addEventListener('resize', () => {
        initCanvas();
        createParticles();
    });

    // Initialize particle system
    if (canvas) {
        initParticleSystem();
    }

    // === THEME TOGGLE ===
    function setTheme(theme) {
        body.classList.remove('light-theme', 'dark-theme');
        body.classList.add(theme);
        localStorage.setItem('preferredTheme', theme);
        
        if (themeToggleButton) {
            themeToggleButton.innerHTML = theme === 'dark-theme' ? 
                '<i class="fas fa-moon"></i>' : '<i class="fas fa-sun"></i>';
        }

        // Update particle colors
        particles.forEach(particle => {
            particle.updateColor();
        });
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const currentTheme = body.classList.contains('dark-theme') ? 'dark-theme' : 'light-theme';
            const newTheme = currentTheme === 'dark-theme' ? 'light-theme' : 'dark-theme';
            setTheme(newTheme);
        });
    }

    // Load preferred theme
    const preferredTheme = localStorage.getItem('preferredTheme');
    setTheme(preferredTheme || 'light-theme');

    // === TEXT CAROUSEL ===
    const carouselMessages = [
        "Phân tích dữ liệu an ninh mạng với AI tiên tiến...",
        "Mô hình machine learning đang học hỏi từng ngày.",
        "Bảo vệ hệ thống khỏi các mối đe dọa tiềm ẩn.",
        "Dự đoán và ngăn chặn rủi ro an ninh mạng.",
        "Công nghệ AI - Tương lai của bảo mật thông tin."
    ];
    let carouselIndex = 0;

    function updateCarouselText() {
        if (carouselTextElement) {
            carouselTextElement.style.opacity = 0;
            setTimeout(() => {
                carouselIndex = (carouselIndex + 1) % carouselMessages.length;
                carouselTextElement.textContent = carouselMessages[carouselIndex];
                carouselTextElement.style.opacity = 1;
            }, 300);
        }
    }

    if (carouselTextElement && carouselMessages.length > 0) {
        carouselTextElement.textContent = carouselMessages[0];
        carouselTextElement.style.opacity = 1;
        setInterval(updateCarouselText, 4000);
    }

    // === PAGE NAVIGATION ===
    function switchPage(currentPage, nextPage, direction = 'forward') {
        if (currentPage) {
            currentPage.classList.add(direction === 'forward' ? 'exit-to-left' : 'exit-to-right');
            currentPage.classList.remove('active-section');
        }

        setTimeout(() => {
            if (currentPage) currentPage.style.display = 'none';
            if (nextPage) {
                nextPage.style.display = 'block';
                nextPage.classList.remove('exit-to-left', 'exit-to-right');
                nextPage.classList.add('active-section');
                void nextPage.offsetWidth;
                nextPage.style.animation = null;
                nextPage.style.animation = direction === 'forward' ? 
                    'fadeInFromRight 0.5s ease-out forwards' : 
                    'fadeInFromLeft 0.5s ease-out forwards';
            }
        }, 400);
    }

    function showMainSelectionPage() {
        currentPredictionOperationId = null;
        switchPage(inputAreaContainer, pageMainSelection, 'backward');
        activeInputType = null;
        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) {
            singlePredictionResult.textContent = '---';
            singlePredictionResult.style.display = 'none';
        }
        if(errorSection) errorSection.style.display = 'none';
        if (featureInput) featureInput.value = '';
        if (fileInput) fileInput.value = '';
    }

    function showDirectInputPage() {
        currentPredictionOperationId = null;
        switchPage(pageMainSelection, inputAreaContainer, 'forward');
        if(directInputSection) directInputSection.style.display = 'block';
        if(fileUploadSection) fileUploadSection.style.display = 'none';
        activeInputType = 'direct';
        if (fileInput) fileInput.value = '';
        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) singlePredictionResult.style.display = 'none';
        if(errorSection) errorSection.style.display = 'none';
    }

    function showFileUploadPage() {
        currentPredictionOperationId = null;
        switchPage(pageMainSelection, inputAreaContainer, 'forward');
        if(directInputSection) directInputSection.style.display = 'none';
        if(fileUploadSection) fileUploadSection.style.display = 'block';
        activeInputType = 'file';
        if (featureInput) featureInput.value = '';
        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) singlePredictionResult.style.display = 'none';
        if(errorSection) errorSection.style.display = 'none';
    }

    // Initial page setup
    if(pageMainSelection) pageMainSelection.classList.add('active-section');
    if(inputAreaContainer) inputAreaContainer.style.display = 'none';

    // Event listeners
    if(btnShowDirectInputPage) btnShowDirectInputPage.addEventListener('click', showDirectInputPage);
    if(btnShowFileUploadPage) btnShowFileUploadPage.addEventListener('click', showFileUploadPage);
    if(backToSelectionButton) backToSelectionButton.addEventListener('click', showMainSelectionPage);

    // === PREDICTION LOGIC ===
    if(predictButton) predictButton.addEventListener('click', async () => {
        const operationId = Date.now();
        currentPredictionOperationId = operationId;

        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) {
            singlePredictionResult.textContent = 'Đang xử lý...';
            singlePredictionResult.style.display = 'block';
            singlePredictionResult.className = 'status-message';
        }
        if(errorSection) errorSection.style.display = 'none';
        if(errorMessage) errorMessage.textContent = '';

        const apiUrl = apiUrlInput ? apiUrlInput.value.trim() : 'http://127.0.0.1:8888/predict';
        if (!apiUrl) {
            showError("Vui lòng nhập URL của API.");
            currentPredictionOperationId = null;
            return;
        }

        let allFeatureLines = [];

        if (activeInputType === 'file') {
            if (fileInput && fileInput.files.length > 0) {
                try {
                    const file = fileInput.files[0];
                    const fileContent = await readFileContent(file);
                    let linesFromFile = fileContent.split('\n').map(line => line.trim()).filter(line => line !== '');
                    if (linesFromFile.length > 0 && isHeaderLikely(linesFromFile[0])) {
                        linesFromFile.shift();
                    }
                    allFeatureLines = linesFromFile;
                    if (allFeatureLines.length === 0) {
                        showError("Tệp CSV không chứa dữ liệu hợp lệ hoặc chỉ chứa dòng tiêu đề.");
                        currentPredictionOperationId = null;
                        return;
                    }
                } catch (error) {
                    showError(`Lỗi khi đọc tệp: ${error.message}`);
                    currentPredictionOperationId = null;
                    return;
                }
            } else {
                showError("Vui lòng chọn một tệp CSV để tải lên.");
                currentPredictionOperationId = null;
                return;
            }
        } else if (activeInputType === 'direct') {
            if (featureInput && featureInput.value.trim() !== '') {
                allFeatureLines = featureInput.value.trim().split('\n').map(line => line.trim()).filter(line => line !== '');
                if (allFeatureLines.length === 0) {
                    showError("Vui lòng nhập dữ liệu đặc trưng hợp lệ.");
                    currentPredictionOperationId = null;
                    return;
                }
            } else {
                showError("Vui lòng nhập dữ liệu đặc trưng vào ô văn bản.");
                currentPredictionOperationId = null;
                return;
            }
        } else {
            showError("Vui lòng chọn một phương thức nhập liệu trước.");
            currentPredictionOperationId = null;
            return;
        }
        
        if(singlePredictionResult) singlePredictionResult.textContent = `Đang xử lý ${allFeatureLines.length} mẫu...`;
        let hasErrors = false;
        let resultsCount = 0;
        let processedCount = 0;

        for (let i = 0; i < allFeatureLines.length; i++) {
            if (currentPredictionOperationId !== operationId) {
                console.log("Quá trình dự đoán đã bị hủy bỏ.");
                if(singlePredictionResult) {
                    singlePredictionResult.textContent = 'Quá trình xử lý đã bị hủy.';
                    singlePredictionResult.style.color = 'var(--text-color-secondary)';
                    singlePredictionResult.style.display = 'block';
                }
                return;
            }

            const line = allFeatureLines[i];
            const parseResult = parseFeatureString(line);
            const featuresArray = parseResult.features; 
            const paddedCount = parseResult.padded;
            const truncatedCount = parseResult.truncated;

            if (parseResult.error) {
                addResultToList(`Mẫu ${i + 1}: ${parseResult.error}`, null, true, 0, 0, []); 
                hasErrors = true;
                processedCount++;
                continue;
            }
            
            try {
                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: featuresArray }),
                });

                resultsCount++;
                const responseText = await response.text();
                let data;
                try {
                    data = JSON.parse(responseText);
                } catch (e) {
                    addResultToList(`Mẫu ${i + 1}: Lỗi phân tích JSON từ API - ${responseText}`, null, true, paddedCount, truncatedCount, featuresArray);
                    hasErrors = true;
                    processedCount++;
                    continue;
                }

                if (!response.ok) {
                    addResultToList(`Mẫu ${i + 1}: Lỗi API ${response.status} - ${data.detail || response.statusText || 'Không có thông tin chi tiết'}`, null, true, paddedCount, truncatedCount, featuresArray);
                    hasErrors = true;
                } else {
                    if (data.prediction) {
                        addResultToList(`Mẫu ${i + 1}: `, data.prediction, false, paddedCount, truncatedCount, featuresArray);
                    } else if (data.error) {
                        addResultToList(`Mẫu ${i + 1}: Lỗi từ API - ${data.error}`, null, true, paddedCount, truncatedCount, featuresArray);
                        hasErrors = true;
                    } else {
                        addResultToList(`Mẫu ${i + 1}: Phản hồi không xác định - ${JSON.stringify(data)}`, null, true, paddedCount, truncatedCount, featuresArray);
                        hasErrors = true;
                    }
                }
            } catch (error) {
                resultsCount++;
                console.error(`Lỗi khi gọi API cho mẫu ${i + 1}:`, error);
                addResultToList(`Mẫu ${i + 1}: Không thể kết nối hoặc có lỗi từ API - ${error.message}`, null, true, paddedCount, truncatedCount, featuresArray);
                hasErrors = true;
            }
            processedCount++;
            
            if (processedCount % 10 === 0 && singlePredictionResult && currentPredictionOperationId === operationId) {
                singlePredictionResult.textContent = `Đang xử lý... ${processedCount}/${allFeatureLines.length} mẫu.`;
            }
        }

        if (currentPredictionOperationId === operationId) {
            if (resultsCount > 0 && singlePredictionResult) {
                singlePredictionResult.style.display = 'none';
            } else if (!hasErrors && allFeatureLines.length > 0 && singlePredictionResult) {
                singlePredictionResult.textContent = 'Không có mẫu nào được xử lý thành công.';
            } else if (!hasErrors && allFeatureLines.length === 0 && singlePredictionResult) {
                singlePredictionResult.textContent = '---';
            }

            if (hasErrors && predictionList && predictionList.children.length === 0 && allFeatureLines.length > 0) {
                showError("Đã xảy ra lỗi trong quá trình xử lý tất cả các mẫu.");
            }
            currentPredictionOperationId = null;
        }
    });

    // === HELPER FUNCTIONS ===
    function isHeaderLikely(line) {
        // Check if line contains letters and doesn't look like pure numeric data
        return /[a-zA-Z]/.test(line) && !/^[0-9.,\sfeE+-]+$/.test(line);
    }

    function parseFeatureString(featureStr) {
        let numberArray;
        try {
            const stringArray = featureStr.split(',');
            if (stringArray.length === 1 && stringArray[0].trim() === "") {
                return { error: "Dòng trống, không có đặc trưng.", features: [], padded: 0, truncated: 0 };
            }
            numberArray = stringArray.map(s => {
                const num = parseFloat(s.trim());
                if (isNaN(num)) {
                    throw new Error(`Giá trị không hợp lệ: '${s.trim()}' không phải là số.`);
                }
                return num;
            });
        } catch (error) {
            console.error("Lỗi parseFeatureString:", error.message);
            return { error: error.message, features: [], padded: 0, truncated: 0 };
        }
        
        let paddedCount = 0;
        let truncatedCount = 0;
        if (numberArray.length < N_EXPECTED_FEATURES) {
            paddedCount = N_EXPECTED_FEATURES - numberArray.length;
            for (let i = 0; i < paddedCount; i++) numberArray.push(0);
        } else if (numberArray.length > N_EXPECTED_FEATURES) {
            truncatedCount = numberArray.length - N_EXPECTED_FEATURES;
            numberArray = numberArray.slice(0, N_EXPECTED_FEATURES);
        }
        return { features: numberArray, padded: paddedCount, truncated: truncatedCount, error: null };
    }

    function readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => resolve(event.target.result);
            reader.onerror = (error) => reject(error);
            reader.readAsText(file);
        });
    }

    function addResultToList(messagePrefix, predictionLabel, isError = false, paddedCount = 0, truncatedCount = 0, originalFeatures = []) {
        if (!predictionList) return;
        const listItem = document.createElement('li');
        let fullMessage = messagePrefix;

        if (!isError && predictionLabel) fullMessage += predictionLabel;
        else if (isError && predictionLabel) fullMessage += predictionLabel;
        
        if (paddedCount > 0) fullMessage += ` (đã thêm ${paddedCount} đặc trưng 0)`;
        if (truncatedCount > 0) fullMessage += ` (đã bỏ ${truncatedCount} đặc trưng thừa)`;
        listItem.textContent = fullMessage;

        if (isError) {
            listItem.style.color = `var(--error-color)`;
            listItem.style.backgroundColor = `rgba(231, 76, 60, 0.1)`;
            listItem.style.borderLeft = `5px solid var(--error-color)`;
            listItem.classList.add('result-item-error');
        } else {
            listItem.style.cursor = 'pointer';
            listItem.classList.add('clickable-result');
            listItem.dataset.features = JSON.stringify(originalFeatures);

            listItem.addEventListener('click', function() {
                if (this.dataset.features && !this.classList.contains('result-item-error')) {
                    try {
                        const features = JSON.parse(this.dataset.features);
                        const featureString = features.map((f, idx) => `F${idx + 1}: ${Number(f.toFixed(4))}`).join('\n');
                        alert(`Đặc trưng đầu vào cho "${predictionLabel}":\n\n${featureString}`);
                    } catch (e) {
                        alert("Không thể truy xuất dữ liệu đặc trưng cho mục này.");
                        console.error("Lỗi khi phân tích cú pháp đặc trưng từ thuộc tính data:", e);
                    }
                }
            });

            const labelForColor = predictionLabel ? predictionLabel.trim().toLowerCase() : "";
            if (labelForColor === 'benign') {
                listItem.style.color = `var(--accent-color-secondary)`;
                listItem.style.backgroundColor = `rgba(46, 204, 113, 0.1)`;
                listItem.style.borderLeft = `5px solid var(--accent-color-secondary)`;
            } else if (labelForColor === 'unknown') {
                listItem.style.color = `var(--error-color)`;
                listItem.style.backgroundColor = `rgba(231, 76, 60, 0.1)`;
                listItem.style.borderLeft = `5px solid var(--error-color)`;
            } else {
                // Specific attack types - Yellow/Warning colors
                listItem.style.color = `var(--text-color-warning)`;
                listItem.style.backgroundColor = `rgba(241, 196, 15, 0.15)`;
                listItem.style.borderLeft = `5px solid var(--accent-color-warning)`;
            }
        }
        predictionList.appendChild(listItem);
    }

    function showError(message) {
        if(errorMessage) errorMessage.textContent = message;
        if(errorSection) errorSection.style.display = 'block';
        if(singlePredictionResult) singlePredictionResult.style.display = 'none';
        currentPredictionOperationId = null;
    }

    // === SCROLL ANIMATIONS ===
    const scrollAnimatedElements = document.querySelectorAll('.results-container');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    scrollAnimatedElements.forEach(el => observer.observe(el));

    // === PERFORMANCE OPTIMIZATIONS ===
    
    // Throttle resize events
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            initCanvas();
            createParticles();
        }, 100);
    });

    // Pause animations when tab is not visible
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        } else {
            if (canvas && ctx) {
                animateParticles();
            }
        }
    });

    // === CLEANUP ===
    window.addEventListener('beforeunload', () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        clearTimeout(moveTimeout);
        clearTimeout(resizeTimeout);
    });

    // === MOBILE DEVICE DETECTION & OPTIMIZATION ===
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    if (isMobile) {
        // Reduce particle count on mobile
        const originalCreateParticles = createParticles;
        createParticles = function() {
            particles = [];
            const particleCount = Math.min(20, Math.floor(window.innerWidth / 30));
            
            for (let i = 0; i < particleCount; i++) {
                particles.push(new EnhancedParticle());
            }
        };
        
        // Disable cursor effects on mobile
        if (cursorTrail) cursorTrail.style.display = 'none';
        if (cursorGlow) cursorGlow.style.display = 'none';
        if (cursorField) cursorField.style.display = 'none';
        if (cursorRipples) cursorRipples.style.display = 'none';
    }

    // === ACCESSIBILITY ENHANCEMENTS ===
    
    // Respect user's motion preferences
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    
    if (prefersReducedMotion) {
        // Disable or reduce animations for users who prefer reduced motion
        document.documentElement.style.setProperty('--transition-speed-fast', '0s');
        document.documentElement.style.setProperty('--transition-speed-normal', '0s');
        document.documentElement.style.setProperty('--transition-speed-slow', '0s');
        
        // Reduce particle count
        if (typeof createParticles === 'function') {
            const originalCreateParticles = createParticles;
            createParticles = function() {
                particles = [];
                const particleCount = Math.min(10, Math.floor(window.innerWidth / 50));
                
                for (let i = 0; i < particleCount; i++) {
                    particles.push(new EnhancedParticle());
                }
            };
        }
    }

    // === KEYBOARD NAVIGATION SUPPORT ===
    document.addEventListener('keydown', (e) => {
        // Add focus styling for keyboard navigation
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });

    document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-navigation');
    });

    console.log('🎨 Enhanced UI with dynamic background and cursor effects loaded successfully!');
});