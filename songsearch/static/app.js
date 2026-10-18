'use strict';
const $ = id => document.getElementById(id);
const labels = new Map();
let catalogue, storageKey, pending, configured = [], evaluationVersion = 0;
const customCases = () => [...labels].map(([query, ids]) => ({query, relevant: [...ids]}));
const songNames = new Map();

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
async function get(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the search inputs.');
  return data;
}
function status(id, text, error = false) {
  $(id).textContent = text;
  $(id).classList.toggle('error', error);
}
function downloadLink(id, data, name) {
  $(id).href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(data, null, 2))}`;
  $(id).download = name;
  $(id).hidden = false;
}
function renderLabels() {
  const cases = customCases();
  $('label-summary').textContent = `Relevance labels (${cases.length} ${cases.length === 1 ? 'query' : 'queries'})`;
  $('label-json').value = JSON.stringify(cases, null, 2);
  $('export-labels').hidden = !cases.length;
  $('evaluate-labels').disabled = !cases.length;
  if ($('eval-source').value === 'custom') showReferences();
  if (cases.length) downloadLink('export-labels', cases, 'judgments.json');
}
function markButton(query, id) {
  const button = node('button');
  const update = () => {
    const selected = labels.get(query)?.has(id) || false;
    button.textContent = selected ? 'Relevant ✓' : 'Mark relevant';
    button.setAttribute('aria-pressed', String(selected));
  };
  update();
  button.onclick = () => {
    const ids = labels.get(query) || new Set();
    if (ids.has(id)) ids.delete(id); else ids.add(id);
    if (ids.size) labels.set(query, ids); else labels.delete(query);
    update(); renderLabels();
    try {
      localStorage.setItem(storageKey, JSON.stringify([...labels].map(([q, ids]) => [q, [...ids]])));
      const selected = new Set(customCases().flatMap(c => c.relevant));
      localStorage.setItem(storageKey + '.songs', JSON.stringify([...songNames].filter(([id]) => selected.has(id))));
    }
    catch { status('search-status', 'Browser storage is unavailable. Download your labels before closing.', true); }
  };
  return button;
}
function renderResults(data) {
  $('results').replaceChildren(...data.results.map(song => {
    songNames.set(song.id, {id: song.id, title: song.title, artist: song.artist});
    const row = node('li', undefined, 'result');
    const content = node('div');
    content.append(node('h2', song.title), node('p', [song.artist, song.genre, song.year].filter(Boolean).join(' · '), 'artist'));
    const words = song.lyrics.trim().split(/\s+/);
    content.append(node('p', song.lyrics ? words.slice(0, 8).join(' ') + (words.length > 8 ? '…' : '') : song.description, 'excerpt'));
    const details = node('details');
    details.append(node('summary', 'Lyrics and scores'));
    details.append(node('p', `Cosine: ${song.semantic_score?.toFixed(3) ?? '—'} · BM25: ${song.keyword_score.toFixed(3)}`, 'scores'));
    if (song.matched_terms.length) details.append(node('p', `Matched words: ${song.matched_terms.join(', ')}`));
    details.append(node('pre', song.lyrics || 'No lyrics supplied. This record uses a description.'));
    if (song.source_url) {
      const source = node('a', 'Source'); source.href = song.source_url;
      source.target = '_blank'; source.rel = 'noopener noreferrer'; details.append(source);
    }
    content.append(details);
    const score = node('aside');
    score.append(node('span', `Score ${song.score.toFixed(4)}`), markButton(data.query, song.id));
    row.append(content, score);
    return row;
  }));
  status('search-status', data.count ? `${data.count} results · ${data.mode} · ${data.elapsed_ms} ms` : 'No results. Try a different description or artist.');
}
$('mode').onchange = () => $('alpha').disabled = $('mode').value !== 'hybrid';
$('search-form').onsubmit = async event => {
  event.preventDefault();
  const q = $('query').value.trim();
  if (!q || !catalogue) return;
  pending?.abort();
  const request = new AbortController(); pending = request;
  $('search-button').disabled = true; status('search-status', 'Searching…');
  const params = new URLSearchParams({q, mode: $('mode').value, artist: $('artist').value.trim(), k: $('limit').value, alpha: $('alpha').value});
  try { renderResults(await get(`/api/search?${params}`, {signal: request.signal})); }
  catch (error) { if (error.name !== 'AbortError') status('search-status', error.message, true); }
  finally { if (pending === request) $('search-button').disabled = false; }
};
for (const view of ['search', 'eval']) $(view + '-tab').onclick = () => {
  $('search-view').hidden = view !== 'search'; $('evaluation-view').hidden = view !== 'eval';
  $('search-tab').setAttribute('aria-pressed', String(view === 'search'));
  $('eval-tab').setAttribute('aria-pressed', String(view === 'eval'));
};
function referenceCases() {
  return $('eval-source').value === 'configured' ? configured : customCases().map(c => ({
    query: c.query, expected: c.relevant.map(id => songNames.get(id) || {id, title: id, artist: 'Saved song'})
  }));
}
function renderCases(cases, configurations = []) {
  for (let i = 0; i < cases.length; i++) {
    const example = node('details', undefined, 'query-case');
    example.append(node('summary', cases[i].query));
    example.append(node('h3', `Expected songs (${cases[i].expected.length})`));
    const expected = node('ol');
    for (const song of cases[i].expected) expected.append(node('li', `${song.title} — ${song.artist}`));
    example.append(expected);
    for (const config of configurations) {
      const result = config.queries[i];
      example.append(node('h3', config.mode));
      example.append(node('p', Object.entries(result).filter(([key]) => key.startsWith('precision@') || key.startsWith('recall@') || key === 'mrr').map(([key, value]) => `${key}: ${value.toFixed(4)}`).join(' · '), 'scores'));
      const list = node('ol');
      for (const song of result.results) list.append(node('li', `${song.title} — ${song.artist}${song.relevant ? ' [labeled relevant]' : ''}`));
      if (!result.results.length) example.append(node('p', 'No matching results.'));
      example.append(list);
    }
    $('evaluation').append(example);
  }
}
function showReferences() {
  evaluationVersion++;
  $('evaluation').replaceChildren(); $('download-report').hidden = true;
  const cases = referenceCases();
  renderCases(cases);
  $('run-eval').disabled = !catalogue || !cases.length;
  status('eval-status', cases.length ? `${cases.length} queries · ${cases.reduce((n, c) => n + c.expected.length, 0)} relevance labels. Expand a query to see expected songs.` : 'Mark search results as relevant to create your test set.');
}
$('eval-source').onchange = showReferences;
$('evaluate-labels').onclick = () => {
  $('eval-source').value = 'custom'; showReferences(); $('eval-tab').click();
  $('eval-form').requestSubmit();
};
$('eval-form').onsubmit = async event => {
  event.preventDefault();
  const version = ++evaluationVersion;
  $('run-eval').disabled = true; status('eval-status', 'Evaluating all queries…');
  try {
    const k = Number($('eval-k').value), alpha = Number($('eval-alpha').value);
    const report = $('eval-source').value === 'custom'
      ? await get('/api/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cases: customCases(), k, alpha})})
      : await get(`/api/evaluate?k=${k}&alpha=${alpha}`);
    if (version !== evaluationVersion) return;
    const table = node('table'), head = node('tr');
    for (const text of ['Method', `Precision@${report.k}`, `Recall@${report.k}`, 'MRR']) head.append(node('th', text));
    const thead = node('thead'); thead.append(head); table.append(thead);
    const tbody = node('tbody');
    for (const config of report.configurations) {
      const row = node('tr'); row.append(node('td', config.mode));
      for (const value of Object.values(config.metrics)) row.append(node('td', value.toFixed(4)));
      tbody.append(row);
    }
    table.append(tbody); $('evaluation').replaceChildren(table);
    renderCases(report.cases, report.configurations);
    status('eval-status', `${report.query_count} queries · ${report.catalogue_size.toLocaleString()} songs · K = ${report.k}`);
    downloadLink('download-report', report, 'evaluation.json');
  } catch (error) { if (version === evaluationVersion) status('eval-status', error.message, true); }
  finally { if (version === evaluationVersion) $('run-eval').disabled = false; }
};
(async () => {
  try {
    catalogue = await get('/api/catalogue');
    for (const option of $('mode').options) option.disabled = !catalogue.modes.includes(option.value);
    if (catalogue.backend === 'keyword') $('mode').value = 'keyword';
    $('mode').onchange();
    storageKey = `songsearch.labels.${catalogue.catalogue_id}`;
    try {
      for (const [id, song] of JSON.parse(localStorage.getItem(storageKey + '.songs') || '[]')) {
        if (song && typeof song.title === 'string' && typeof song.artist === 'string') songNames.set(id, song);
      }
      for (const [query, ids] of JSON.parse(localStorage.getItem(storageKey) || '[]')) {
        if (typeof query === 'string' && Array.isArray(ids) && ids.every(id => typeof id === 'string')) labels.set(query, new Set(ids));
      }
    } catch { /* Search remains usable if browser storage is unavailable. */ }
    configured = await get('/api/judgments');
    for (const c of configured) for (const song of c.expected) songNames.set(song.id, song);
    $('eval-source').options[0].disabled = !configured.length;
    if (!configured.length) $('eval-source').value = 'custom';
    renderLabels(); showReferences(); $('search-button').disabled = false;
    const count = catalogue.count.toLocaleString();
    $('catalogue-info').textContent = `${count} songs in catalogue`;
    status('search-status', `Ready. Search ${count} songs.`);
  } catch (error) { status('search-status', `Could not load the catalogue: ${error.message}`, true); }
})();
