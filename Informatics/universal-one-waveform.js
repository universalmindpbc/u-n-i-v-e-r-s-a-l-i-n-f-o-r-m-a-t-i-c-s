// Microphone waveform and palette reused from the I Am voice assistant.
(() => {
 window.UniversalOneWaveform=(root,canvases,inputEnabled)=>{
 let waveformAudioContext=null,waveformSource=null,waveformAnalyser=null,waveformFrame=0,waveformSamples=null,waveformLevel=0;
  function startMicrophoneWaveform(stream) {
    stopMicrophoneWaveform();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !canvases.length) return;
    try {
      waveformAudioContext = new AudioContextClass();
      waveformSource = waveformAudioContext.createMediaStreamSource(stream);
      waveformAnalyser = waveformAudioContext.createAnalyser();
      waveformAnalyser.fftSize = 256;
      waveformAnalyser.smoothingTimeConstant = 0.72;
      waveformSamples = new Uint8Array(waveformAnalyser.fftSize);
      waveformSource.connect(waveformAnalyser);
      waveformAudioContext.resume?.().catch(() => {});
      waveformFrame = window.requestAnimationFrame(drawWaveformFrame);
    } catch {
      stopMicrophoneWaveform();
    }
  }

  function stopMicrophoneWaveform() {
    if (waveformFrame) window.cancelAnimationFrame(waveformFrame);
    waveformFrame = 0;
    try { waveformSource?.disconnect(); } catch {}
    waveformSource = null;
    waveformAnalyser = null;
    waveformSamples = null;
    waveformLevel = 0;
    root.dataset.inputActive = "false";
    const context = waveformAudioContext;
    waveformAudioContext = null;
    context?.close?.().catch(() => {});
    drawWaveformIdle();
  }

  function drawWaveformIdle() {
    drawWaveformLine({ state: root.dataset.state || "idle", level: 0, samples: null, time: 0 });
  }

  function drawWaveformFrame(time) {
    const state = root.dataset.state || "idle";
    const receivesInput = state === "listening" && inputEnabled();
    let samples = null;
    let level = 0;

    if (receivesInput && waveformAnalyser && waveformSamples) {
      waveformAnalyser.getByteTimeDomainData(waveformSamples);
      let energy = 0;
      for (let index = 0; index < waveformSamples.length; index += 1) {
        const sample = (waveformSamples[index] - 128) / 128;
        energy += sample * sample;
      }
      level = Math.sqrt(energy / waveformSamples.length);
      waveformLevel = waveformLevel * 0.68 + level * 0.32;
      samples = waveformSamples;
    } else {
      waveformLevel *= 0.72;
    }

    root.dataset.inputActive = String(receivesInput && waveformLevel > 0.018);
    drawWaveformLine({ state, level: waveformLevel, samples, time });
    waveformFrame = window.requestAnimationFrame(drawWaveformFrame);
  }

  function drawWaveformLine({ state, level, samples, time }) {
    for (const waveformCanvas of canvases) {
    const context = waveformCanvas.getContext("2d");
    if (!context) continue;
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;
    const centre = height / 2;
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    context.clearRect(0, 0, width, height);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = 3;
    context.strokeStyle = state === "speaking"
      ? "rgba(228, 190, 123, .96)"
      : (state === "muted" ? "rgba(210, 220, 219, .52)" : "rgba(117, 223, 213, .96)");
    context.beginPath();

    const startX = 10;
    const endX = width - 10;
    const span = endX - startX;
    const hasLiveInput = samples && !reducedMotion;
    for (let point = 0; point <= 28; point += 1) {
      const ratio = point / 28;
      const x = startX + ratio * span;
      let y = centre;
      if (hasLiveInput) {
        const sampleIndex = Math.min(samples.length - 1, Math.floor(ratio * samples.length));
        const sample = (samples[sampleIndex] - 128) / 128;
        const boost = Math.min(4.8, 1.45 + level * 22);
        y += sample * height * 0.29 * boost;
      } else if (state === "speaking" && !reducedMotion) {
        y += Math.sin(ratio * Math.PI * 4 + time / 330) * 3.2;
      } else if (state !== "muted") {
        y += Math.sin(ratio * Math.PI * 4) * 1.15;
      }
      if (point === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();

    if (state === "muted") {
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(21, 43);
      context.lineTo(43, 21);
      context.stroke();
    }
    }
  }

 return {start:startMicrophoneWaveform,stop:stopMicrophoneWaveform,idle:drawWaveformIdle,getState:()=>({active:!!waveformAnalyser,level:waveformLevel})};
 };
})();
