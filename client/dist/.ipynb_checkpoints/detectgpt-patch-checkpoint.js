// detectgpt-patch.js - UPDATED VERSION WITH FASTDETECT
(function() {
    'use strict';
    
    // DetectGPT UI Management
    class DetectGPTManager {
        constructor() {
            this.container = null;
            this.initialized = false;
        }
        
        init() {
            if (this.initialized) return;
            
            // Add CSS styles
            this.addStyles();
            
            // Create toggle control
            this.createToggle();
            
            this.initialized = true;
        }
        
        addStyles() {
            const style = document.createElement('style');
            style.textContent = `
                .detectgpt-container {
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 16px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .detectgpt-container h3 {
                    margin-top: 0;
                    margin-bottom: 12px;
                    color: #495057;
                    font-size: 18px;
                    border-bottom: 1px solid #dee2e6;
                    padding-bottom: 8px;
                }
                
                .detection-methods {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 16px;
                }
                
                .detection-method {
                    flex: 1;
                    background: #ffffff;
                    border: 1px solid #e9ecef;
                    border-radius: 6px;
                    padding: 12px;
                }
                
                .method-title {
                    font-weight: bold;
                    margin-bottom: 8px;
                    color: #343a40;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                .detectgpt-score {
                    font-size: 16px;
                    margin-bottom: 12px;
                }
                
                .detectgpt-score .label {
                    font-weight: bold;
                    color: #343a40;
                }
                
                .detectgpt-score .score-value {
                    font-size: 20px;
                    font-weight: bold;
                    margin: 0 8px;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
                
                .detectgpt-score .score-value.high-ai { 
                    color: #dc3545; 
                    background: rgba(220, 53, 69, 0.1);
                }
                
                .detectgpt-score .score-value.medium-ai { 
                    color: #fd7e14;
                    background: rgba(253, 126, 20, 0.1);
                }
                
                .detectgpt-score .score-value.low-ai { 
                    color: #28a745;
                    background: rgba(40, 167, 69, 0.1);
                }
                
                .detectgpt-score .score-interpretation {
                    font-size: 14px;
                    font-style: italic;
                    color: #6c757d;
                }
                
                .detectgpt-details {
                    font-size: 14px;
                    color: #6c757d;
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 12px;
                    margin-top: 8px;
                }
                
                .detectgpt-details div {
                    margin: 4px 0;
                }
                
                .detectgpt-details span {
                    font-weight: 500;
                    color: #495057;
                }
                
                .fastdetect-details {
                    font-size: 14px;
                    color: #6c757d;
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 12px;
                    margin-top: 8px;
                }
                
                .fastdetect-details div {
                    margin: 4px 0;
                }
                
                .fastdetect-details span {
                    font-weight: 500;
                    color: #495057;
                }
                
                .error-message {
                    background: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                    border-radius: 4px;
                    padding: 8px 12px;
                    margin-top: 12px;
                    font-size: 14px;
                }
                
                .detection-toggles {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                    margin: 8px 0;
                }
                
                .detectgpt-toggle {
                    display: flex;
                    align-items: center;
                    font-size: 14px;
                    color: #495057;
                    cursor: pointer;
                }
                
                .detectgpt-toggle input[type="checkbox"] {
                    margin-right: 8px;
                    cursor: pointer;
                }
                
                .detectgpt-toggle span {
                    cursor: pointer;
                    user-select: none;
                }
                
                .detectgpt-toggle:hover {
                    color: #007bff;
                }
                
                .api-key-input {
                    margin-top: 8px;
                    margin-left: 24px;
                    display: none;
                }
                
                .api-key-input input {
                    width: 200px;
                    padding: 4px 8px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    font-size: 12px;
                }
                
                .api-key-input label {
                    display: block;
                    font-size: 12px;
                    margin-bottom: 4px;
                    color: #6c757d;
                }
                
                .method-unavailable {
                    color: #6c757d;
                    font-style: italic;
                    font-size: 14px;
                }
            `;
            document.head.appendChild(style);
        }
        
        createToggle() {
            // Find a good place to insert the toggle
            const inputGroup = document.querySelector('.input-group') || 
                             document.querySelector('#submit_text_btn').parentElement;
            
            if (inputGroup) {
                const toggleDiv = document.createElement('div');
                toggleDiv.className = 'control-group';
                toggleDiv.innerHTML = `
                    <div class="detection-toggles">
                        <label class="detectgpt-toggle">
                            <input type="checkbox" id="enable_detectgpt" checked>
                            <span>Enable DetectGPT Analysis</span>
                        </label>
                        <label class="detectgpt-toggle">
                            <input type="checkbox" id="enable_fastdetect">
                            <span>Enable FastDetect Analysis</span>
                        </label>
                        <div class="api-key-input" id="fastdetect_api_key_input">
                            <label for="fastdetect_api_key">FastDetect API Key:</label>
                            <input type="password" id="fastdetect_api_key" placeholder="Enter API key">
                        </div>
                    </div>
                `;
                inputGroup.appendChild(toggleDiv);
                
                // Show/hide API key input based on FastDetect toggle
                const fastdetectToggle = document.getElementById('enable_fastdetect');
                const apiKeyInput = document.getElementById('fastdetect_api_key_input');
                
                fastdetectToggle.addEventListener('change', function() {
                    apiKeyInput.style.display = this.checked ? 'block' : 'none';
                });
            }
        }
        
        createContainer() {
            if (this.container) return this.container;
            
            const resultsArea = document.querySelector('#all_result') || 
                              document.querySelector('#results').parentElement;
            
            const container = document.createElement('div');
            container.id = 'detectgpt_results';
            container.className = 'detectgpt-container';
            container.style.display = 'none';
            container.innerHTML = `
                <h3>AI Detection Analysis</h3>
                <div class="detection-methods">
                    <div class="detection-method" id="detectgpt_method">
                        <div class="method-title">DetectGPT</div>
                        <div class="detectgpt-score">
                            <span class="label">Score:</span>
                            <span id="detectgpt_score" class="score-value"></span>
                            <div class="score-interpretation" id="detectgpt_interpretation"></div>
                        </div>
                        <div class="detectgpt-details">
                            <div>Original Likelihood: <span id="original_ll"></span></div>
                            <div>Average Perturbed Likelihood: <span id="avg_perturbed_ll"></span></div>
                            <div>Perturbations: <span id="n_perturbations"></span></div>
                        </div>
                        <div id="detectgpt_error" class="error-message" style="display: none;"></div>
                    </div>
                    
                    <div class="detection-method" id="fastdetect_method">
                        <div class="method-title">FastDetect</div>
                        <div class="detectgpt-score">
                            <span class="label">Probability:</span>
                            <span id="fastdetect_prob" class="score-value"></span>
                            <div class="score-interpretation" id="fastdetect_interpretation"></div>
                        </div>
                        <div class="fastdetect-details">
                            <div>Detection Metric (CRIT): <span id="fastdetect_crit"></span></div>
                            <div>Tokens Processed: <span id="fastdetect_tokens"></span></div>
                        </div>
                        <div id="fastdetect_error" class="error-message" style="display: none;"></div>
                    </div>
                </div>
            `;
            
            if (resultsArea) {
                resultsArea.insertBefore(container, resultsArea.firstChild);
            }
            
            this.container = container;
            return container;
        }
        
        updateResults(detectgptData, fastdetectData) {
            const container = this.createContainer();
            
            let hasResults = false;
            
            // Update DetectGPT results
            if (detectgptData && detectgptData.detectgpt_score !== null && detectgptData.detectgpt_score !== undefined) {
                hasResults = true;
                document.querySelector('#detectgpt_method').style.display = 'block';
                document.querySelector('#detectgpt_error').style.display = 'none';
                
                // Update score
                const scoreElement = document.querySelector('#detectgpt_score');
                scoreElement.textContent = detectgptData.detectgpt_score.toFixed(3);
                
                // Add interpretation and styling based on score
                const interpretation = document.querySelector('#detectgpt_interpretation');
                const score = detectgptData.detectgpt_score;
                
                // Remove existing classes
                scoreElement.className = 'score-value';
                
                if (score > 1.0) {
                    scoreElement.classList.add('high-ai');
                    interpretation.textContent = '(Likely AI-generated)';
                } else if (score > 0.5) {
                    scoreElement.classList.add('medium-ai');
                    interpretation.textContent = '(Possibly AI-generated)';
                } else {
                    scoreElement.classList.add('low-ai');
                    interpretation.textContent = '(Likely human-written)';
                }
                
                // Update details
                document.querySelector('#original_ll').textContent = 
                    detectgptData.original_ll ? detectgptData.original_ll.toFixed(3) : 'N/A';
                document.querySelector('#avg_perturbed_ll').textContent = 
                    detectgptData.avg_perturbed_ll ? detectgptData.avg_perturbed_ll.toFixed(3) : 'N/A';
                document.querySelector('#n_perturbations').textContent = 
                    detectgptData.n_perturbations || 0;
                    
            } else if (detectgptData && detectgptData.error) {
                hasResults = true;
                document.querySelector('#detectgpt_method').style.display = 'block';
                document.querySelector('#detectgpt_error').style.display = 'block';
                document.querySelector('#detectgpt_error').textContent = 
                    `DetectGPT Error: ${detectgptData.error}`;
            } else {
                document.querySelector('#detectgpt_method').style.display = 'none';
            }
            
            // Update FastDetect results
            if (fastdetectData && fastdetectData.success && fastdetectData.prob !== null && fastdetectData.prob !== undefined) {
                hasResults = true;
                document.querySelector('#fastdetect_method').style.display = 'block';
                document.querySelector('#fastdetect_error').style.display = 'none';
                
                // Update probability
                const probElement = document.querySelector('#fastdetect_prob');
                const prob = fastdetectData.prob;
                probElement.textContent = (prob * 100).toFixed(1) + '%';
                
                // Add interpretation and styling based on probability
                const interpretation = document.querySelector('#fastdetect_interpretation');
                
                // Remove existing classes
                probElement.className = 'score-value';
                
                if (prob > 0.7) {
                    probElement.classList.add('high-ai');
                    interpretation.textContent = '(Likely AI-generated)';
                } else if (prob > 0.4) {
                    probElement.classList.add('medium-ai');
                    interpretation.textContent = '(Possibly AI-generated)';
                } else {
                    probElement.classList.add('low-ai');
                    interpretation.textContent = '(Likely human-written)';
                }
                
                // Update details
                document.querySelector('#fastdetect_crit').textContent = 
                    fastdetectData.crit ? fastdetectData.crit.toFixed(3) : 'N/A';
                document.querySelector('#fastdetect_tokens').textContent = 
                    fastdetectData.ntoken || 'N/A';
                    
            } else if (fastdetectData && !fastdetectData.success) {
                hasResults = true;
                document.querySelector('#fastdetect_method').style.display = 'block';
                document.querySelector('#fastdetect_error').style.display = 'block';
                document.querySelector('#fastdetect_error').textContent = 
                    `FastDetect Error: ${fastdetectData.error}`;
            } else {
                document.querySelector('#fastdetect_method').style.display = 'none';
            }
            
            // Show/hide container based on whether we have any results
            container.style.display = hasResults ? 'block' : 'none';
        }
        
        isDetectGPTEnabled() {
            const toggle = document.querySelector('#enable_detectgpt');
            return toggle ? toggle.checked : true;
        }
        
        isFastDetectEnabled() {
            const toggle = document.querySelector('#enable_fastdetect');
            return toggle ? toggle.checked : false;
        }
        
        getFastDetectApiKey() {
            const keyInput = document.querySelector('#fastdetect_api_key');
            return keyInput ? keyInput.value.trim() : '';
        }
    }
    
    // Patch the original API function instead of replacing event handlers
    window.addEventListener('load', function() {
        const detectGPTManager = new DetectGPTManager();
        
        // Wait for the original script to initialize
        setTimeout(() => {
            // Initialize DetectGPT UI
            detectGPTManager.init();
            
            // Hook into the existing $ function (which handles API responses)
            // Find the original $ function in the global scope or window object
            const originalDollarFunction = window.$ || window.updateFromRequest;
            
            if (typeof originalDollarFunction === 'function') {
                // Create enhanced version that includes DetectGPT
                window.$ = function(data) {
                    // Call original function first
                    const result = originalDollarFunction.call(this, data);
                    
                    // Add DetectGPT and FastDetect handling
                    if (data && data.result) {
                        detectGPTManager.updateResults(
                            data.result.detectgpt,
                            data.result.fastdetect
                        );
                    }
                    
                    return result;
                };
            } else {
                // Fallback: patch the API analyze function
                const originalFetch = window.fetch;
                window.fetch = function(url, options) {
                    if (url.includes('/api/analyze') && options && options.method === 'POST') {
                        // Modify the request to include DetectGPT and FastDetect
                        const body = JSON.parse(options.body);
                        
                        if (detectGPTManager.isDetectGPTEnabled()) {
                            body.include_detectgpt = true;
                        }
                        
                        if (detectGPTManager.isFastDetectEnabled()) {
                            const apiKey = detectGPTManager.getFastDetectApiKey();
                            if (apiKey) {
                                body.include_fastdetect = true;
                                body.fastdetect_api_key = apiKey;
                            }
                        }
                        
                        options.body = JSON.stringify(body);
                    }
                    
                    return originalFetch.apply(this, arguments).then(response => {
                        if (url.includes('/api/analyze') && response.ok) {
                            return response.json().then(data => {
                                // Handle DetectGPT and FastDetect results
                                if (data && data.result) {
                                    detectGPTManager.updateResults(
                                        data.result.detectgpt,
                                        data.result.fastdetect
                                    );
                                }
                                return new Response(JSON.stringify(data), {
                                    status: response.status,
                                    statusText: response.statusText,
                                    headers: response.headers
                                });
                            });
                        }
                        return response;
                    });
                };
            }
        }, 2000); // Wait for original initialization
    });
    
    // Export for global access
    window.DetectGPTManager = DetectGPTManager;
})();