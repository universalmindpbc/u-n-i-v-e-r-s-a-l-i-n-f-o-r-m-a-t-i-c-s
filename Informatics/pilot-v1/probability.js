'use strict';
(() => {
  const section = document.querySelector('#atlas-probability');
  const dialog = document.createElement('dialog');
  dialog.className = 'science-reader probability-reader';
  dialog.setAttribute('aria-labelledby', 'probability-reader-title');
  document.body.append(dialog);
  let opener;
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const formatCount = value => String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

  dialog.addEventListener('close', () => {
    if (opener?.isConnected && !section.hidden) opener.focus();
  });
  dialog.addEventListener('click', event => {
    if (event.target.closest('[data-probability-close]')) dialog.close();
    if (event.target === dialog) {
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    }
  });

  window.AtlasProbability = {
    setRecord(record) {
      section.hidden = true;
      if (dialog.open) dialog.close();
      section.replaceChildren();
      dialog.replaceChildren();
      const p = record.probability;
      if (!p || p.status !== 'Calculated' || p.record_id !== record.id) return;
      section.innerHTML = `<h3 id="probability-heading">${escape(p.heading)}</h3><p class="probability-value">${escape(p.display)}</p><p class="probability-summary">${escape(p.summary)}</p><button type="button" class="probability-open" aria-haspopup="dialog">View calculation →</button>${p.research_insight?.quote ? `<blockquote class="probability-insight"><p>“${escape(p.research_insight.quote)}”</p></blockquote>` : ""}`;
      dialog.innerHTML = `<div class="science-reader-bar"><span>M${record.modulus} / P${record.pisano_period}</span><button type="button" class="science-close" data-probability-close aria-label="Close probability calculation">×</button></div><div class="science-reader-body"><h2 id="probability-reader-title">${escape(p.heading)}</h2><p class="probability-value">${escape(p.display)}</p><ol class="probability-steps">${p.calculation.steps.map(step => `<li>${escape(step)}</li>`).join('')}</ol><h3>Calculation</h3><dl class="probability-counts"><dt>Possible arrangements</dt><dd>${formatCount(p.counts.all_arrangements)}</dd><dt>Matching arrangements</dt><dd>${formatCount(p.counts.matching_arrangements)}</dd></dl><p class="probability-formula">${escape(p.calculation.formula)}</p><p>${escape(p.calculation.counting_note)}</p><details class="probability-verification"><summary>Verification</summary><p>${escape(p.calculation.verification_note)}</p><p>Calculated ${escape(p.calculated_on)} · original Gamma and Beta alignment preserved.</p></details><div class="probability-downloads"><a href="pilot-v1/${encodeURIComponent(p.files.data)}?v=m9-m18-probability-v1" download>Calculation JSON ↗</a><a href="pilot-v1/${encodeURIComponent(p.files.code)}?v=m9-m18-probability-v1" download>Python calculation ↗</a></div></div>`;
      section.querySelector('button').addEventListener('click', event => {
        opener = event.currentTarget;
        (document.fullscreenElement || document.body).append(dialog);
        dialog.showModal();
        dialog.scrollTop = 0;
      });
      section.hidden = false;
    },
  };
})();
