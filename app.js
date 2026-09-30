const form = document.getElementById('patientForm');
const fileInput = document.getElementById('xray');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const statusBox = document.getElementById('status');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
});

async function compressImage(file) {
  const bitmap = await createImageBitmap(file);
  const maxSide = 1600;
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.82));
  if (!blob) throw new Error('Could not compress the image.');
  return blob;
}

function esc(value) {
  return String(value).replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#039;'}[c]));
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    statusBox.textContent = 'Please choose a chest X-ray.';
    return;
  }

  analyzeBtn.disabled = true;
  statusBox.textContent = 'Compressing image and running multimodal analysis…';
  results.hidden = true;

  try {
    const image = await compressImage(file);
    const fd = new FormData();
    fd.append('image', image, 'xray.jpg');
    for (const id of ['age','sex','temperature','spo2','wbc','neutrophils','lymphocytes']) {
      fd.append(id, document.getElementById(id).value);
    }

    const response = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Analysis failed.');

    renderResult(data, URL.createObjectURL(image));
    statusBox.textContent = 'Analysis complete.';
  } catch (error) {
    statusBox.textContent = error.message || 'Something went wrong.';
  } finally {
    analyzeBtn.disabled = false;
  }
});

function renderResult(data, originalUrl) {
  results.hidden = false;
  const score = Number(data.screening.score);
  document.getElementById('overall').innerHTML = `
    <div class="muted">${esc(data.disease)} screening</div>
    <div class="score">${score.toFixed(2)}%</div>
    <strong>${esc(data.screening.label)} model score</strong>
  `;

  document.getElementById('xrayText').innerHTML = `
    <div><strong>${esc(data.xray.assessment)}</strong></div>
    <div class="muted">Image model score: ${data.xray.pneumonia_score}% • uncertainty: ${data.xray.uncertainty}%</div>
  `;
  document.getElementById('originalResult').src = originalUrl;
  document.getElementById('gradcam').src = data.xray.gradcam;

  document.getElementById('selectedFeatures').innerHTML = data.clinical.ga_selected_features
    .map(f => `<span class="feature">${esc(f)}</span>`).join('');

  const values = data.clinical.input;
  document.getElementById('clinicalTable').innerHTML = `
    <table>${Object.entries(values).map(([k,v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')}</table>
  `;

  const shap = data.clinical.shap;
  const maxAbs = Math.max(...shap.map(x => Math.abs(x.shap)), 1e-9);
  document.getElementById('shap').innerHTML = shap.map(x => {
    const width = Math.round((Math.abs(x.shap) / maxAbs) * 100);
    const sign = x.shap >= 0 ? '+' : '−';
    return `<div class="shap-row"><span>${esc(x.feature)}</span><div class="bar"><i style="width:${width}%"></i></div><b>${sign}${Math.abs(x.shap).toFixed(3)}</b></div>`;
  }).join('');

  document.getElementById('fusionText').innerHTML = `
    Image representation: <strong>${data.fusion.image_representation_dimensions}D</strong><br>
    Clinical representation: <strong>${data.fusion.clinical_representation_dimensions}D</strong><br>
    Prediction network: <strong>${esc(data.fusion.prediction_network)}</strong>
  `;
}
