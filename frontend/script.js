
document.addEventListener('DOMContentLoaded', () => {
    // --- Existing DOM Elements ---
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

    // --- New DOM Elements ---
    const themeToggleButton = document.getElementById('themeToggleButton');
    const carouselTextElement = document.getElementById('carouselText');
    const body = document.body;

    const N_EXPECTED_FEATURES = 67;
    if(expectedFeaturesCountSpan) expectedFeaturesCountSpan.textContent = N_EXPECTED_FEATURES;
    let activeInputType = null;
    let currentPredictionOperationId = null; // To track and cancel ongoing predictions

    // --- Theme Toggle ---
    function setTheme(theme) {
        body.classList.remove('light-theme', 'dark-theme');
        body.classList.add(theme);
        localStorage.setItem('preferredTheme', theme);
        if (themeToggleButton) {
            themeToggleButton.innerHTML = theme === 'dark-theme' ? '<i class="fas fa-moon"></i>' : '<i class="fas fa-sun"></i>';
        }
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const currentTheme = body.classList.contains('dark-theme') ? 'dark-theme' : 'light-theme';
            const newTheme = currentTheme === 'dark-theme' ? 'light-theme' : 'dark-theme';
            setTheme(newTheme);
        });
    }
    // Load preferred theme or default to light
    const preferredTheme = localStorage.getItem('preferredTheme');
    setTheme(preferredTheme || 'light-theme');


    // --- Text Carousel ---
    const carouselMessages = [
        "Phân tích dữ liệu an ninh mạng...",
        "Mô hình AI đang học hỏi mỗi ngày.",
        "Bảo vệ hệ thống của bạn.",
        "Dự đoán rủi ro tiềm ẩn.",
        "An toàn là ưu tiên hàng đầu."
    ];
    let carouselIndex = 0;
    function updateCarouselText() {
        if (carouselTextElement) {
            carouselTextElement.style.opacity = 0;
            setTimeout(() => {
                carouselIndex = (carouselIndex + 1) % carouselMessages.length;
                carouselTextElement.textContent = carouselMessages[carouselIndex];
                carouselTextElement.style.opacity = 1;
            }, 300); // Match CSS transition
        }
    }
    if (carouselTextElement && carouselMessages.length > 0) {
        carouselTextElement.textContent = carouselMessages[0]; // Initial text
        carouselTextElement.style.opacity = 1;
        setInterval(updateCarouselText, 5000); // Change text every 5 seconds
    }


    // --- Page Navigation Functions with Animation ---
    function switchPage(currentPage, nextPage, direction = 'forward') {
        if (currentPage) {
            currentPage.classList.add(direction === 'forward' ? 'exit-to-left' : 'exit-to-right');
            currentPage.classList.remove('active-section');
        }

        setTimeout(() => {
            if (currentPage) currentPage.style.display = 'none';
            if (nextPage) {
                nextPage.style.display = 'block';
                nextPage.classList.remove('exit-to-left', 'exit-to-right'); // Clear any exit classes
                nextPage.classList.add('active-section');
                // Trigger reflow for animation restart if needed, though class switching should be enough
                void nextPage.offsetWidth; 
                nextPage.style.animation = null; // Clear previous animation
                nextPage.style.animation = direction === 'forward' ? 'fadeInFromRight 0.5s ease-out forwards' : 'fadeInFromLeft 0.5s ease-out forwards';
            }
        }, 400); // Match CSS animation duration
    }

    function showMainSelectionPage() {
        currentPredictionOperationId = null; // Signal to cancel any ongoing prediction
        switchPage(inputAreaContainer, pageMainSelection, 'backward');
        activeInputType = null;
        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) {
            singlePredictionResult.textContent = '---'; // Reset status message
            singlePredictionResult.style.display = 'none';
        }
        if(errorSection) errorSection.style.display = 'none';
        // Clear inputs
        if (featureInput) featureInput.value = '';
        if (fileInput) fileInput.value = '';
    }

    function showDirectInputPage() {
        currentPredictionOperationId = null; // Cancel previous if any, though less likely here
        switchPage(pageMainSelection, inputAreaContainer, 'forward');
        if(directInputSection) directInputSection.style.display = 'block';
        if(fileUploadSection) fileUploadSection.style.display = 'none';
        activeInputType = 'direct';
        if (fileInput) fileInput.value = ''; // Clear other input
        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) singlePredictionResult.style.display = 'none';
        if(errorSection) errorSection.style.display = 'none';
    }

    function showFileUploadPage() {
        currentPredictionOperationId = null; // Cancel previous if any
        switchPage(pageMainSelection, inputAreaContainer, 'forward');
        if(directInputSection) directInputSection.style.display = 'none';
        if(fileUploadSection) fileUploadSection.style.display = 'block';
        activeInputType = 'file';
        if (featureInput) featureInput.value = ''; // Clear other input
        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) singlePredictionResult.style.display = 'none';
        if(errorSection) errorSection.style.display = 'none';
    }
    
    // Initial page setup
    if(pageMainSelection) pageMainSelection.classList.add('active-section');
    if(inputAreaContainer) inputAreaContainer.style.display = 'none';


    // --- Event Listeners for Page Navigation ---
    if(btnShowDirectInputPage) btnShowDirectInputPage.addEventListener('click', showDirectInputPage);
    if(btnShowFileUploadPage) btnShowFileUploadPage.addEventListener('click', showFileUploadPage);
    if(backToSelectionButton) backToSelectionButton.addEventListener('click', showMainSelectionPage);

    // --- Prediction Logic (largely unchanged, ensure elements exist) ---
    if(predictButton) predictButton.addEventListener('click', async () => {
        const operationId = Date.now(); // Generate a unique ID for this prediction operation
        currentPredictionOperationId = operationId; // Set this as the active operation

        if(predictionList) predictionList.innerHTML = '';
        if(singlePredictionResult) {
            singlePredictionResult.textContent = 'Đang xử lý...';
            singlePredictionResult.style.display = 'block';
            singlePredictionResult.className = 'status-message'; // Reset class
        }
        if(errorSection) errorSection.style.display = 'none';
        if(errorMessage) errorMessage.textContent = '';

        const apiUrl = apiUrlInput ? apiUrlInput.value.trim() : 'http://127.0.0.1:8888/predict'; // Default if input not found
        if (!apiUrl) {
            showError("Vui lòng nhập URL của API.");
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
                        currentPredictionOperationId = null; // Clear operation on early exit
                        return;
                    }
                } catch (error) {
                    showError(`Lỗi khi đọc tệp: ${error.message}`);
                    currentPredictionOperationId = null; // Clear operation on early exit
                    return;
                }
            } else {
                showError("Vui lòng chọn một tệp CSV để tải lên.");
                currentPredictionOperationId = null; // Clear operation on early exit
                return;
            }
        } else if (activeInputType === 'direct') {
            if (featureInput && featureInput.value.trim() !== '') {
                allFeatureLines = featureInput.value.trim().split('\n').map(line => line.trim()).filter(line => line !== '');
                if (allFeatureLines.length === 0) {
                    showError("Vui lòng nhập dữ liệu đặc trưng hợp lệ.");
                    currentPredictionOperationId = null; // Clear operation on early exit
                    return;
                }
            } else {
                showError("Vui lòng nhập dữ liệu đặc trưng vào ô văn bản.");
                currentPredictionOperationId = null; // Clear operation on early exit
                return;
            }
        } else {
            showError("Vui lòng chọn một phương thức nhập liệu trước.");
            currentPredictionOperationId = null; // Clear operation on early exit
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
                    singlePredictionResult.style.color = 'var(--text-color-secondary)'; // Or a specific warning color
                    singlePredictionResult.style.display = 'block';
                }
                // Do not clear predictionList here, as some results might have been populated
                // and the user might want to see them before they are cleared on next navigation.
                return; // Exit the loop and function
            }

            const line = allFeatureLines[i];
            // ... (rest of the loop: parseResult, API call, addResultToList) ...
            // Ensure featuresArray is defined before API call
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

                resultsCount++; // Counts successful or attempted API calls
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
             // Update progress if needed, e.g., every few items or if the list is very long
            if (processedCount % 10 === 0 && singlePredictionResult && currentPredictionOperationId === operationId) {
                singlePredictionResult.textContent = `Đang xử lý... ${processedCount}/${allFeatureLines.length} mẫu.`;
            }
        }
        // This part only runs if the loop completed naturally (not cancelled)
        if (currentPredictionOperationId === operationId) {
            if (resultsCount > 0 && singlePredictionResult) {
                singlePredictionResult.style.display = 'none'; // Hide "Đang xử lý..." if results are shown
            } else if (!hasErrors && allFeatureLines.length > 0 && singlePredictionResult) {
                singlePredictionResult.textContent = 'Không có mẫu nào được xử lý thành công.';
            } else if (!hasErrors && allFeatureLines.length === 0 && singlePredictionResult) {
                singlePredictionResult.textContent = '---'; // Back to default if no lines
            }

            if (hasErrors && predictionList && predictionList.children.length === 0 && allFeatureLines.length > 0) {
                showError("Đã xảy ra lỗi trong quá trình xử lý tất cả các mẫu.");
            }
            currentPredictionOperationId = null; // Mark operation as complete
        }
    });

    // --- Helper Functions (parseFeatureString, readFileContent, addResultToList, showError, isHeaderLikely) ---
    // These are largely the same, ensure they use the correct CSS variables for colors if needed
    function isHeaderLikely(line) {
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

        const currentThemeSuffix = body.classList.contains('dark-theme') ? 'dark' : 'light';

        if (isError) {
            listItem.style.color = `var(--error-color-${currentThemeSuffix}, var(--error-color))`;
            listItem.style.backgroundColor = `color-mix(in srgb, var(--error-color-${currentThemeSuffix}, var(--error-color)) 15%, transparent)`;
            listItem.style.borderLeft = `5px solid var(--error-color-${currentThemeSuffix}, var(--error-color))`;
            listItem.classList.add('result-item-error'); // Add class for error items
        } else {
            listItem.style.cursor = 'pointer';
            listItem.classList.add('clickable-result');
            listItem.dataset.features = JSON.stringify(originalFeatures); // Store features

            listItem.addEventListener('click', function() {
                if (this.dataset.features && !this.classList.contains('result-item-error')) {
                    try {
                        const features = JSON.parse(this.dataset.features);
                        const featureString = features.map((f, idx) => `F${idx + 1}: ${Number(f.toFixed(4))}`).join('\n'); // Format numbers
                        alert(`Đặc trưng đầu vào cho "${predictionLabel}":\n\n${featureString}`);
                    } catch (e) {
                        alert("Không thể truy xuất dữ liệu đặc trưng cho mục này.");
                        console.error("Lỗi khi phân tích cú pháp đặc trưng từ thuộc tính data:", e);
                    }
                }
            });

            const labelForColor = predictionLabel ? predictionLabel.trim().toLowerCase() : "";
            if (labelForColor === 'benign') {
                listItem.style.color = `var(--accent-color-secondary-${currentThemeSuffix}, var(--accent-color-secondary))`;
                listItem.style.backgroundColor = `color-mix(in srgb, var(--accent-color-secondary-${currentThemeSuffix}, var(--accent-color-secondary)) 15%, transparent)`;
                listItem.style.borderLeft = `5px solid var(--accent-color-secondary-${currentThemeSuffix}, var(--accent-color-secondary))`;
            } else if (labelForColor === 'unknown') {
                listItem.style.color = `var(--error-color-${currentThemeSuffix}, var(--error-color))`;
                listItem.style.backgroundColor = `color-mix(in srgb, var(--error-color-${currentThemeSuffix}, var(--error-color)) 15%, transparent)`;
                listItem.style.borderLeft = `5px solid var(--error-color-${currentThemeSuffix}, var(--error-color))`;
            } else { // Specific attack types - NOW YELLOW
                listItem.style.color = `var(--text-color-warning-${currentThemeSuffix}, var(--text-color-warning))`;
                listItem.style.backgroundColor = `color-mix(in srgb, var(--accent-color-warning-${currentThemeSuffix}, var(--accent-color-warning)) 25%, transparent)`;
                listItem.style.borderLeft = `5px solid var(--accent-color-warning-${currentThemeSuffix}, var(--accent-color-warning))`;
            }
        }
        predictionList.appendChild(listItem);
    }

    function showError(message) {
        if(errorMessage) errorMessage.textContent = message;
        if(errorSection) errorSection.style.display = 'block';
        if(singlePredictionResult) singlePredictionResult.style.display = 'none';
    }

    // --- Scroll Animations ---
    const scrollAnimatedElements = document.querySelectorAll('.results-container'); // Add more selectors if needed

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // observer.unobserve(entry.target); // Optional: unobserve after animation
            } else {
                // Optional: remove 'visible' if you want animation to replay on scroll up
                // entry.target.classList.remove('visible'); 
            }
        });
    }, { threshold: 0.1 }); // Trigger when 10% of the element is visible

    scrollAnimatedElements.forEach(el => observer.observe(el));

});