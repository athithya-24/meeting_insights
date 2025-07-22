let mediaRecorder;
let audioChunks = [];
let startTime;
let timerInterval;
let currentMeetingData = null;
let audioContext;
let analyser;
let audioLevel;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
  // Set up UI elements
  document.getElementById('stats').style.display = 'none';
  document.getElementById('transcriptActions').style.display = 'none';
  document.getElementById('summaryActions').style.display = 'none';
  
  // Check for microphone permissions
  checkMicrophonePermission();
  
  // Load meeting history
  loadMeetingHistory();
  
  // Set up event listeners
  setupEventListeners();
  
  // Initialize audio level visualization
  initializeAudioLevel();
});

function checkMicrophonePermission() {
  if (navigator.permissions) {
    navigator.permissions.query({ name: 'microphone' }).then(result => {
      if (result.state === 'denied') {
        showAlert('Microphone access is required for this application to work. Please enable microphone permissions in your browser settings.', 'warning');
      }
    }).catch(err => {
      console.error('Permission query failed:', err);
    });
  }
}

function setupEventListeners() {
  // Start recording
  document.getElementById('startBtn').addEventListener('click', startRecording);
  
  // Stop recording
  document.getElementById('stopBtn').addEventListener('click', stopRecording);
  document.getElementById('vadStartBtn').addEventListener('click', startVAD);
  document.getElementById('vadStopBtn').addEventListener('click', stopVAD);
  
  // Periodically check for VAD updates
  setInterval(checkVADUpdates, 2000);
}

// VAD Control Functions
async function startVAD() {
  try {
    showAlert('Starting voice detection...', 'info');
    updateVADUI(true, 'starting');
    
    const response = await fetch('/vad/start', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    const data = await response.json();
    showAlert(`Voice detection started (Session: ${data.session_id})`, 'success');
    updateVADUI(true, 'active');
    document.getElementById('vadResults').style.display = 'block';
    
  } catch (error) {
    console.error('Error starting VAD:', error);
    showAlert(`Failed to start voice detection: ${error.message}`, 'error');
    updateVADUI(false, 'error');
  }
}

async function stopVAD() {
  try {
    updateVADUI(true, 'stopping');
    
    const response = await fetch('/vad/stop', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    const data = await response.json();
    showAlert('Voice detection stopped', 'success');
    updateVADUI(false, 'ready');
    
  } catch (error) {
    console.error('Error stopping VAD:', error);
    showAlert(`Failed to stop voice detection: ${error.message}`, 'error');
    updateVADUI(false, 'error');
  }
}

function updateVADUI(isActive, status) {
  const startBtn = document.getElementById('vadStartBtn');
  const stopBtn = document.getElementById('vadStopBtn');
  const statusText = document.getElementById('vadStatusText');
  const statusIndicator = document.getElementById('vadStatusIndicator');
  
  // Reset classes
  statusIndicator.className = 'status-indicator';
  
  switch (status) {
    case 'starting':
      statusIndicator.classList.add('processing');
      statusText.textContent = 'Starting...';
      startBtn.disabled = true;
      stopBtn.disabled = true;
      break;
      
    case 'active':
      statusIndicator.classList.add('recording');
      statusText.textContent = 'Active - Listening...';
      startBtn.disabled = true;
      stopBtn.disabled = false;
      break;
      
    case 'stopping':
      statusIndicator.classList.add('processing');
      statusText.textContent = 'Processing...';
      startBtn.disabled = true;
      stopBtn.disabled = true;
      break;
      
    case 'error':
      statusIndicator.classList.add('error');
      statusText.textContent = 'Error occurred';
      startBtn.disabled = false;
      stopBtn.disabled = true;
      break;
      
    default: // ready
      statusIndicator.classList.add('ready');
      statusText.textContent = 'Ready';
      startBtn.disabled = false;
      stopBtn.disabled = true;
  }
}

async function checkVADUpdates() {
  try {
    // Only check if VAD is active
    const vadStatus = document.getElementById('vadStatusIndicator');
    if (!vadStatus || !vadStatus.classList.contains('recording')) {
      return;
    }
    
    const response = await fetch('/vad/status');
    if (!response.ok) return;
    
    const status = await response.json();
    
    // Process any new results
    if (status.latest_results && status.latest_results.length > 0) {
      const transcriptContainer = document.getElementById('realTimeTranscript');
      
      status.latest_results.forEach(result => {
        if (result.transcript) {
          const transcriptDiv = document.createElement('div');
          transcriptDiv.className = 'transcript-segment';
          transcriptDiv.innerHTML = `
            <div class="transcript-text">${result.transcript}</div>
            <div class="transcript-meta">
              ${new Date().toLocaleTimeString()}
            </div>
          `;
          transcriptContainer.appendChild(transcriptDiv);
        }
      });
      
      // Auto-scroll to bottom
      transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
    }
    
  } catch (error) {
    console.error('Error checking VAD updates:', error);
  }

  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
      const tabName = this.getAttribute('onclick').match(/'([^']+)'/)[1];
      switchTab(tabName);
    });
  });
}

function initializeAudioLevel() {
  audioLevel = document.getElementById('audioLevel');
  if (audioLevel) {
    audioLevel.style.width = '0%';
  }
}

function startRecording() {
  navigator.mediaDevices.getUserMedia({ 
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      sampleRate: 44100
    }
  }).then(stream => {
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    audioChunks = [];
    startTime = Date.now();
    
    // Check audio quality
    checkAudioQuality(stream);
    
    // Update UI
    updateRecordingUI(true);
    
    // Start timer
    timerInterval = setInterval(updateTimer, 1000);
    
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };
    
    mediaRecorder.onstop = async () => {
      clearInterval(timerInterval);
      if (audioContext) {
        audioContext.close();
      }
      updateRecordingUI(false, 'processing');
      await processRecording();
    };
    
    mediaRecorder.onerror = (e) => {
      console.error('MediaRecorder error:', e);
      showAlert('Recording error occurred. Please try again.', 'error');
      updateRecordingUI(false);
    };
    
    mediaRecorder.start(100); // Collect data every 100ms
    
  }).catch(error => {
    console.error('Error accessing microphone:', error);
    let errorMessage = 'Error accessing microphone. ';
    if (error.name === 'NotAllowedError') {
      errorMessage += 'Please allow microphone access and try again.';
    } else if (error.name === 'NotFoundError') {
      errorMessage += 'No microphone found. Please check your audio devices.';
    } else {
      errorMessage += 'Please check permissions and try again.';
    }
    showAlert(errorMessage, 'error');
  });
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    // Stop all tracks to release microphone
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
  }
}

function updateRecordingUI(isRecording, status = 'ready') {
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const timer = document.getElementById('timer');
  
  // Reset classes
  statusIndicator.className = 'status-indicator';
  timer.className = 'timer';
  
  if (isRecording) {
    statusIndicator.classList.add('recording');
    statusText.textContent = 'Recording...';
    startBtn.disabled = true;
    stopBtn.disabled = false;
    timer.classList.add('active');
  } else {
    if (status === 'processing') {
      statusIndicator.classList.add('processing');
      statusText.textContent = 'Processing...';
      startBtn.disabled = true;
      stopBtn.disabled = true;
    } else {
      statusIndicator.classList.add('ready');
      statusText.textContent = 'Ready to record';
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const minutes = Math.floor(elapsed / 60);
  const seconds = elapsed % 60;
  document.getElementById('timer').textContent = 
    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

function checkAudioQuality(stream) {
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(stream);
    
    microphone.connect(analyser);
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const checkLevel = () => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        
        // Update audio level indicator
        if (audioLevel) {
          const level = Math.min((average / 128) * 100, 100);
          audioLevel.style.width = `${level}%`;
          
          // Add visual feedback for audio level
          if (level > 20) {
            audioLevel.style.backgroundColor = '#4CAF50';
          } else if (level > 10) {
            audioLevel.style.backgroundColor = '#FFA726';
          } else {
            audioLevel.style.backgroundColor = '#F44336';
          }
        }
        
        requestAnimationFrame(checkLevel);
      }
    };
    
    checkLevel();
  } catch (error) {
    console.error('Audio quality check failed:', error);
  }
}

async function processRecording() {
  showLoadingStates(true);
  
  try {
    if (audioChunks.length === 0) {
      throw new Error('No audio data recorded');
    }
    
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    
    // Check if blob has data
    if (blob.size === 0) {
      throw new Error('Empty audio recording');
    }
    
    const formData = new FormData();
    formData.append('audio_data', blob, 'meeting_recording.webm');
    
    const response = await fetch('/process', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    currentMeetingData = result;
    
    // Update UI with results
    updateResultsUI(result);
    
    // Show stats
    updateStatsUI(result);
    
    // Switch to transcript tab
    switchTab('transcript');
    
  } catch (error) {
    console.error('Error processing audio:', error);
    showAlert(`Error processing audio: ${error.message}`, 'error');
    
    // Clear loading states and show error
    document.getElementById('transcript').innerHTML = 
      `<div class="empty-state">
        <i class="fas fa-exclamation-triangle" style="color: var(--danger);"></i>
        <p style="color: var(--danger);">Error processing audio: ${error.message}</p>
        <p>Please try recording again.</p>
      </div>`;
  } finally {
    showLoadingStates(false);
    updateRecordingUI(false);
    loadMeetingHistory(); // Refresh history
  }
}

function updateStatsUI(result) {
  document.getElementById('stats').style.display = 'grid';
  document.getElementById('wordCount').textContent = result.word_count || 0;
  document.getElementById('duration').textContent = `${Math.round(result.metadata?.duration || 0)}s`;
  
  // Calculate sentiment score
  const sentimentScore = result.detailed_sentiment?.average_score || 0;
  document.getElementById('sentimentScore').textContent = sentimentScore.toFixed(2);
  
  // Count action items
  const actionItemsText = result.action_items || '';
  const actionItemsCount = actionItemsText.includes('No specific action items') ? 0 : 
    (actionItemsText.match(/\n-|\n\d+\.|•/g) || []).length;
  document.getElementById('actionItems').textContent = actionItemsCount;
}

function updateResultsUI(result) {
  // Update transcript
  const transcriptElement = document.getElementById('transcript');
  transcriptElement.innerHTML = `<p>${formatTranscript(result.transcript)}</p>`;
  document.getElementById('transcriptActions').style.display = 'flex';
  
  // Update summary
  const summaryElement = document.getElementById('summary');
  summaryElement.innerHTML = `
    <div class="insight-card">
      <h4><i class="fas fa-clipboard-check"></i> Meeting Summary</h4>
      <p>${formatText(result.summary)}</p>
    </div>
    <div class="insight-card">
      <h4><i class="fas fa-tasks"></i> Action Items</h4>
      <p>${formatText(result.action_items || 'No specific action items identified.')}</p>
    </div>
    <div class="insight-card">
      <h4><i class="fas fa-smile"></i> Sentiment Analysis</h4>
      <p>Overall sentiment: <span class="sentiment-badge sentiment-${result.detailed_sentiment?.overall || 'neutral'}">
        ${result.detailed_sentiment?.overall || 'neutral'}
      </span></p>
      <p>${formatText(result.sentiment_analysis || 'No sentiment analysis available.')}</p>
    </div>
  `;
  document.getElementById('summaryActions').style.display = 'flex';
  
  // Update insights
  updateInsightsUI(result);
}

function formatTranscript(transcript) {
  if (!transcript) return 'No transcript available.';
  return transcript.replace(/\n/g, '<br>').replace(/\. /g, '. <br><br>');
}

function formatText(text) {
  if (!text) return 'No content available.';
  return text.replace(/\n/g, '<br>');
}

function updateInsightsUI(result) {
  const insightsElement = document.getElementById('insights');
  
  let insightsHTML = `
    <div class="insight-grid">
      <div class="insight-card">
        <h4><i class="fas fa-chart-pie"></i> Word Frequency</h4>
        <div class="word-frequency">
          ${result.word_frequency && Object.keys(result.word_frequency).length > 0 ? 
            Object.entries(result.word_frequency).slice(0, 15).map(([word, count]) => `
              <div class="word-frequency-item">
                <span class="word">${word}</span>
                <span class="word-frequency-count">${count}</span>
              </div>
            `).join('') : '<p>No word frequency data available</p>'}
        </div>
      </div>
      
      <div class="insight-card">
        <h4><i class="fas fa-cloud"></i> Word Cloud</h4>
        <div class="word-cloud-container">
          ${result.word_cloud ? 
            `<img src="${result.word_cloud}" alt="Word cloud" style="max-width: 100%; height: auto;">` : 
            '<p>Word cloud not available</p>'}
        </div>
      </div>
    </div>
    
    <div class="insight-card">
      <h4><i class="fas fa-chart-bar"></i> Sentiment Over Time</h4>
      <div class="sentiment-chart-container">
        ${result.sentiment_chart ? 
          `<img src="${result.sentiment_chart}" alt="Sentiment chart" style="max-width: 100%; height: auto;">` : 
          '<p>Sentiment chart not available</p>'}
      </div>
    </div>
    
    <div class="insight-card">
      <h4><i class="fas fa-clock"></i> Time-Based Summaries</h4>
      <div class="time-segments">
        ${result.time_based_summaries && result.time_based_summaries.length > 0 ? 
          result.time_based_summaries.map(segment => `
            <div class="time-segment">
              <div class="time-segment-header">
                <span><strong>Segment ${segment.segment}</strong> (${formatTime(segment.start_time)} - ${formatTime(segment.end_time)})</span>
                <span class="word-count">${segment.word_count} words</span>
              </div>
              <p>${segment.summary}</p>
            </div>
          `).join('') : '<p>No time-based summaries available</p>'}
      </div>
    </div>
    
    <div class="insight-card">
      <h4><i class="fas fa-file-export"></i> Export Options</h4>
      <div class="export-options">
        <button class="btn btn-secondary" onclick="exportContent('full', 'pdf')">
          <i class="fas fa-file-pdf"></i> Full Report (PDF)
        </button>
        <button class="btn btn-secondary" onclick="exportContent('full', 'docx')">
          <i class="fas fa-file-word"></i> Full Report (DOCX)
        </button>
        <button class="btn btn-secondary" onclick="exportContent('srt', 'srt')">
          <i class="fas fa-file-alt"></i> Subtitles (SRT)
        </button>
      </div>
    </div>
  `;
  
  insightsElement.innerHTML = insightsHTML;
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

function showLoadingStates(show) {
  const transcriptLoading = document.getElementById('transcriptLoading');
  const summaryLoading = document.getElementById('summaryLoading');
  
  if (transcriptLoading) {
    transcriptLoading.style.display = show ? 'flex' : 'none';
  }
  if (summaryLoading) {
    summaryLoading.style.display = show ? 'flex' : 'none';
  }
}

function switchTab(tabName) {
  // Remove active class from all tabs and content
  document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  
  // Add active class to clicked tab
  const clickedTab = Array.from(document.querySelectorAll('.tab')).find(tab => 
    tab.getAttribute('onclick') && tab.getAttribute('onclick').includes(tabName)
  );
  if (clickedTab) {
    clickedTab.classList.add('active');
  }
  
  // Show corresponding content
  const tabContent = document.getElementById(`${tabName}-tab`);
  if (tabContent) {
    tabContent.classList.add('active');
  }
  
  // Special handling for history tab
  if (tabName === 'history') {
    loadFullMeetingHistory();
  }
}

function copyToClipboard(elementId) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  const text = element.textContent || element.innerText;
  
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => {
      showAlert('Copied to clipboard!', 'success');
    }).catch(err => {
      console.error('Failed to copy:', err);
      fallbackCopyTextToClipboard(text);
    });
  } else {
    fallbackCopyTextToClipboard(text);
  }
}

function fallbackCopyTextToClipboard(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.top = "0";
  textArea.style.left = "0";
  textArea.style.position = "fixed";
  
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  
  try {
    const successful = document.execCommand('copy');
    if (successful) {
      showAlert('Copied to clipboard!', 'success');
    } else {
      showAlert('Failed to copy text', 'error');
    }
  } catch (err) {
    console.error('Fallback copy failed:', err);
    showAlert('Copy not supported', 'error');
  }
  
  document.body.removeChild(textArea);
}

function downloadText(elementId, filename) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  const text = element.textContent || element.innerText;
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'download.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function exportContent(contentType, format) {
  if (!currentMeetingData) {
    showAlert('No meeting data available to export', 'error');
    return;
  }
  
  try {
    let content, title;
    
    if (contentType === 'transcript') {
      content = currentMeetingData.transcript;
      title = 'Meeting Transcript';
    } else if (contentType === 'summary') {
      content = `${currentMeetingData.summary}\n\nAction Items:\n${currentMeetingData.action_items}`;
      title = 'Meeting Summary';
    } else if (contentType === 'srt') {
      content = currentMeetingData.srt_subtitles;
      title = 'Meeting Subtitles';
    } else if (contentType === 'full') {
      // Create comprehensive report
      content = `Meeting Report - ${new Date().toLocaleDateString()}

TRANSCRIPT:
${currentMeetingData.transcript}

SUMMARY:
${currentMeetingData.summary}

ACTION ITEMS:
${currentMeetingData.action_items}

SENTIMENT ANALYSIS:
${currentMeetingData.sentiment_analysis}

STATISTICS:
- Words: ${currentMeetingData.word_count}
- Duration: ${Math.round(currentMeetingData.metadata?.duration || 0)} seconds
- Overall Sentiment: ${currentMeetingData.detailed_sentiment?.overall || 'neutral'}`;
      
      title = 'Full Meeting Report';
    }
    
    if (!content) {
      showAlert('No content available to export', 'error');
      return;
    }
    
    const exportUrl = `/export/${format}/${contentType}?content=${encodeURIComponent(content)}&title=${encodeURIComponent(title)}`;
    
    const response = await fetch(exportUrl);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || 'Export failed');
    }
    
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    
    const timestamp = new Date().toISOString().split('T')[0];
    a.download = `meeting_${contentType}_${timestamp}.${format}`;
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showAlert('Export completed successfully!', 'success');
    
  } catch (error) {
    console.error('Export error:', error);
    showAlert(`Failed to export content: ${error.message}`, 'error');
  }
}

async function loadMeetingHistory() {
  try {
    const response = await fetch('/history');
    if (!response.ok) throw new Error('Failed to load history');
    
    const data = await response.json();
    renderMeetingHistory(data.history);
    
  } catch (error) {
    console.error('Error loading history:', error);
    showAlert('Failed to load meeting history', 'error');
  }
}

function renderMeetingHistory(meetings) {
  const historyList = document.getElementById('historyList');
  if (!historyList) return;
  
  if (!meetings || meetings.length === 0) {
    historyList.innerHTML = '<p>No meetings recorded yet.</p>';
    return;
  }
  
  // Show only recent meetings in sidebar (max 5)
  const recentMeetings = meetings.slice(0, 5);
  
  historyList.innerHTML = recentMeetings.map(meeting => `
    <div class="history-item" onclick="loadMeetingDetails('${meeting.filename}')">
      <div class="history-date">
        ${formatDate(meeting.timestamp)}
      </div>
      <div class="history-stats">
        <span>${meeting.word_count} words</span>
        <span>${Math.round(meeting.duration)}s</span>
      </div>
    </div>
  `).join('');
  
  // Show history section
  const historySection = document.getElementById('historySection');
  if (historySection && meetings.length > 0) {
    historySection.style.display = 'block';
  }
}

async function loadFullMeetingHistory() {
  try {
    const response = await fetch('/history');
    if (!response.ok) throw new Error('Failed to load history');
    
    const data = await response.json();
    renderFullMeetingHistory(data.history);
    
  } catch (error) {
    console.error('Error loading full history:', error);
    const historyContent = document.getElementById('historyContent');
    if (historyContent) {
      historyContent.innerHTML = `
        <div class="empty-state">
          <i class="fas fa-exclamation-triangle"></i>
          <p>Failed to load meeting history</p>
        </div>
      `;
    }
  }
}

function renderFullMeetingHistory(meetings) {
  const historyListFull = document.getElementById('historyListFull');
  if (!historyListFull) return;
  
  if (!meetings || meetings.length === 0) {
    historyListFull.innerHTML = `
      <div class="empty-state">
        <i class="fas fa-history"></i>
        <p>No meetings recorded yet. Start recording to see your meeting history here.</p>
      </div>
    `;
    return;
  }
  
  historyListFull.innerHTML = `
    <div class="history-grid">
      ${meetings.map(meeting => `
        <div class="history-card" onclick="loadMeetingDetails('${meeting.filename}')">
          <div class="history-card-header">
            <h4>${formatDate(meeting.timestamp)}</h4>
          </div>
          <div class="history-card-stats">
            <div class="stat">
              <i class="fas fa-file-alt"></i>
              <span>${meeting.word_count} words</span>
            </div>
            <div class="stat">
              <i class="fas fa-clock"></i>
              <span>${formatDuration(meeting.duration)}</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

async function loadMeetingDetails(filename) {
  try {
    const response = await fetch(`/history/${filename}`);
    if (!response.ok) throw new Error('Failed to load meeting details');
    
    const meetingData = await response.json();
    showMeetingModal(meetingData);
    
  } catch (error) {
    console.error('Error loading meeting details:', error);
    showAlert('Failed to load meeting details', 'error');
  }
}

function showMeetingModal(meetingData) {
  const modal = document.getElementById('historyModal');
  const modalTitle = document.getElementById('modalTitle');
  const modalContent = document.getElementById('modalContent');
  
  if (!modal || !modalTitle || !modalContent) return;
  
  modalTitle.textContent = `Meeting - ${formatDate(meetingData.timestamp)}`;
  
  modalContent.innerHTML = `
    <div class="modal-section">
      <h4><i class="fas fa-file-alt"></i> Transcript</h4>
      <p class="modal-text">${formatText(meetingData.transcript)}</p>
    </div>
    
    <div class="modal-section">
      <h4><i class="fas fa-clipboard-list"></i> Summary</h4>
      <p class="modal-text">${formatText(meetingData.summary)}</p>
    </div>
    
    <div class="modal-section">
      <h4><i class="fas fa-chart-bar"></i> Statistics</h4>
      <div class="modal-stats">
        <div class="modal-stat">
          <span class="stat-label">Words:</span>
          <span class="stat-value">${meetingData.word_count}</span>
        </div>
        <div class="modal-stat">
          <span class="stat-label">Duration:</span>
          <span class="stat-value">${formatDuration(meetingData.duration)}</span>
        </div>
      </div>
    </div>
    
    <div class="modal-actions">
      <button class="btn btn-secondary" onclick="copyMeetingData('${meetingData.filename}')">
        <i class="fas fa-copy"></i> Copy Transcript
      </button>
      <button class="btn btn-secondary" onclick="downloadMeetingData('${meetingData.filename}')">
        <i class="fas fa-download"></i> Download
      </button>
    </div>
  `;
  
  modal.style.display = 'flex';
  
  // Close modal when clicking outside
  modal.onclick = function(event) {
    if (event.target === modal) {
      closeModal();
    }
  };
}

function closeModal() {
  const modal = document.getElementById('historyModal');
  if (modal) {
    modal.style.display = 'none';
  }
}

async function copyMeetingData(filename) {
  try {
    const response = await fetch(`/history/${filename}`);
    if (!response.ok) throw new Error('Failed to load meeting data');
    
    const meetingData = await response.json();
    
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(meetingData.transcript);
      showAlert('Meeting transcript copied to clipboard!', 'success');
    } else {
      fallbackCopyTextToClipboard(meetingData.transcript);
    }
    
  } catch (error) {
    console.error('Error copying meeting data:', error);
    showAlert('Failed to copy meeting data', 'error');
  }
}

async function downloadMeetingData(filename) {
  try {
    const response = await fetch(`/history/${filename}`);
    if (!response.ok) throw new Error('Failed to load meeting data');
    
    const meetingData = await response.json();
    
    const content = `Meeting Report - ${formatDate(meetingData.timestamp)}

TRANSCRIPT:
${meetingData.transcript}

SUMMARY:
${meetingData.summary}

METADATA:
- Words: ${meetingData.word_count}
- Duration: ${formatDuration(meetingData.duration)}
- Recorded: ${formatDate(meetingData.timestamp)}`;
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meeting_${formatDate(meetingData.timestamp, true)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
  } catch (error) {
    console.error('Error downloading meeting data:', error);
    showAlert('Failed to download meeting data', 'error');
  }
}

function formatDate(dateString, forFilename = false) {
  const date = new Date(dateString);
  if (forFilename) {
    return date.toISOString().split('T')[0];
  }
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  } else {
    return `${remainingSeconds}s`;
  }
}

function showAlert(message, type = 'info') {
  // Create alert element
  const alert = document.createElement('div');
  alert.className = `alert alert-${type}`;
  alert.innerHTML = `
    <div class="alert-content">
      <span>${message}</span>
      <button class="alert-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
    </div>
  `;
  
  // Add to container
  const alertContainer = document.getElementById('alertContainer');
  if (!alertContainer) {
    // Create container if it doesn't exist
    const container = document.createElement('div');
    container.id = 'alertContainer';
    container.className = 'alert-container';
    document.body.appendChild(container);
    container.appendChild(alert);
  } else {
    alertContainer.appendChild(alert);
  }
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    alert.remove();
    // Remove container if no alerts left
    const alertContainer = document.getElementById('alertContainer');
    if (alertContainer && alertContainer.children.length === 0) {
      alertContainer.remove();
    }
  }, 5000);
}

// Helper function to format text with line breaks and paragraphs
function formatTextWithLineBreaks(text) {
  if (!text) return '';
  return text.replace(/\n/g, '<br>').replace(/\n\n/g, '<p></p>');
}

// Helper function to format action items
function formatActionItems(text) {
  if (!text) return 'No action items identified';
  return text.replace(/\n/g, '<br>').replace(/- /g, '• ');
}

// Helper function to format sentiment score with color
function formatSentimentScore(score) {
  if (score > 0.3) {
    return `<span style="color: #4CAF50;">${score.toFixed(2)} (Positive)</span>`;
  } else if (score < -0.3) {
    return `<span style="color: #F44336;">${score.toFixed(2)} (Negative)</span>`;
  } else {
    return `<span style="color: #FFA726;">${score.toFixed(2)} (Neutral)</span>`;
  }
}

// Helper function to create word frequency chart
function createWordFrequencyChart(wordFrequency) {
  if (!wordFrequency || Object.keys(wordFrequency).length === 0) {
    return '<p>No word frequency data available</p>';
  }
  
  const words = Object.keys(wordFrequency);
  const counts = Object.values(wordFrequency);
  
  // Sort by frequency
  const sortedIndices = [...Array(words.length).keys()].sort((a, b) => counts[b] - counts[a]);
  
  let html = '<div class="word-frequency-chart">';
  sortedIndices.slice(0, 15).forEach(i => {
    const percentage = (counts[i] / Math.max(...counts)) * 100;
    html += `
      <div class="word-frequency-item">
        <span class="word">${words[i]}</span>
        <div class="frequency-bar-container">
          <div class="frequency-bar" style="width: ${percentage}%"></div>
          <span class="count">${counts[i]}</span>
        </div>
      </div>
    `;
  });
  html += '</div>';
  
  return html;
}

// Helper function to format time segments
function formatTimeSegments(segments) {
  if (!segments || segments.length === 0) {
    return '<p>No time segments available</p>';
  }
  
  return segments.map(segment => `
    <div class="time-segment">
      <div class="time-segment-header">
        <span><strong>${segment.start_time} - ${segment.end_time}</strong></span>
        <span class="word-count">${segment.word_count} words</span>
      </div>
      <p>${segment.summary}</p>
    </div>
  `).join('');
}

// Helper function to create sentiment timeline
function createSentimentTimeline(segments) {
  if (!segments || segments.length === 0) {
    return '<p>No sentiment data available</p>';
  }
  
  let html = '<div class="sentiment-timeline">';
  segments.forEach(segment => {
    const sentiment = segment.sentiment_score;
    let color, emoji;
    
    if (sentiment > 0.3) {
      color = '#4CAF50';
      emoji = '😊';
    } else if (sentiment < -0.3) {
      color = '#F44336';
      emoji = '😞';
    } else {
      color = '#FFA726';
      emoji = '😐';
    }
    
    const height = Math.min(Math.abs(sentiment) * 100, 80);
    
    html += `
      <div class="sentiment-item">
        <div class="sentiment-bar" style="height: ${height}px; background-color: ${color};"></div>
        <div class="sentiment-time">${formatTime(segment.time)}</div>
        <div class="sentiment-emoji">${emoji}</div>
      </div>
    `;
  });
  html += '</div>';
  
  return html;
}

// Helper function to format participant analysis
function formatParticipantAnalysis(participants) {
  if (!participants || participants.length === 0) {
    return '<p>No participant data available</p>';
  }
  
  return `
    <div class="participant-grid">
      ${participants.map(participant => `
        <div class="participant-card">
          <h4>${participant.name}</h4>
          <p><strong>Speaking Time:</strong> ${participant.speaking_time}</p>
          <p><strong>Contribution:</strong> ${participant.contribution}</p>
          <p><strong>Sentiment:</strong> ${formatSentimentScore(participant.sentiment)}</p>
        </div>
      `).join('')}
    </div>
  `;
}

// Helper function to create export options
function createExportOptions() {
  return `
    <div class="export-buttons">
      <button class="btn btn-secondary" onclick="exportContent('full', 'pdf')">
        <i class="fas fa-file-pdf"></i> Full Report (PDF)
      </button>
      <button class="btn btn-secondary" onclick="exportContent('full', 'docx')">
        <i class="fas fa-file-word"></i> Full Report (DOCX)
      </button>
      <button class="btn btn-secondary" onclick="exportContent('transcript', 'txt')">
        <i class="fas fa-file-alt"></i> Transcript (TXT)
      </button>
      <button class="btn btn-secondary" onclick="exportContent('summary', 'txt')">
        <i class="fas fa-file-alt"></i> Summary (TXT)
      </button>
      <button class="btn btn-secondary" onclick="exportContent('srt', 'srt')">
        <i class="fas fa-closed-captioning"></i> Subtitles (SRT)
      </button>
    </div>
  `;
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Set up UI elements
  document.getElementById('stats').style.display = 'none';
  document.getElementById('transcriptActions').style.display = 'none';
  document.getElementById('summaryActions').style.display = 'none';
  
  // Check for microphone permissions
  checkMicrophonePermission();
  
  // Load meeting history
  loadMeetingHistory();
  
  // Set up event listeners
  setupEventListeners();
  
  // Initialize audio level visualization
  initializeAudioLevel();
  
  // Set current date in header
  const currentDate = new Date();
  const dateString = currentDate.toLocaleDateString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  });
  document.getElementById('currentDate').textContent = dateString;
});

// Close modal when pressing Escape key
document.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') {
    closeModal();
  }
});

// Prevent form submission
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', function(e) {
    e.preventDefault();
  });
});

// Handle window resize for responsive layout
window.addEventListener('resize', function() {
  // Adjust layout elements if needed
  const audioLevel = document.getElementById('audioLevel');
  if (audioLevel) {
    audioLevel.style.width = '0%';
  }
});

// Handle beforeunload to warn about unsaved recordings
window.addEventListener('beforeunload', function(e) {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    e.preventDefault();
    e.returnValue = 'You have an active recording. Are you sure you want to leave?';
    return e.returnValue;
  }
});

// Add any additional utility functions here
function debounce(func, wait, immediate) {
  let timeout;
  return function() {
    const context = this, args = arguments;
    const later = function() {
      timeout = null;
      if (!immediate) func.apply(context, args);
    };
    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func.apply(context, args);
  };
}

function throttle(func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}