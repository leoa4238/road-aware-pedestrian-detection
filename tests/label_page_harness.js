// Runs the <script> of index_attr.html in Node with a minimal fake DOM and checks the review flow.
// Usage: node tests/label_page_harness.js <index_attr.html>   (prints JSON with the observed results)
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const KEY = /KEY='([^']+)'/.exec(html)[1];
const script = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'));

function el() {
  return { style: {}, classList: { toggle() {} }, dataset: {}, value: '', textContent: '', src: '',
           addEventListener() {}, focus() {}, blur() {} };
}
const els = {}, handlers = {};
let downloaded = null;
const store = {};
global.localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
global.document = {
  getElementById: id => (els[id] = els[id] || el()),
  querySelectorAll: () => [],
  activeElement: null,
  addEventListener: (type, fn) => { handlers[type] = fn; },
  createElement: () => ({ click() {}, set download(v) {}, set href(v) {} }),
};
global.URL = { createObjectURL: blob => { downloaded = blob.text; return 'blob:'; } };
global.Blob = function (parts) { this.text = parts.join(''); };
global.alert = () => {};

function load(prestore) {
  Object.keys(store).forEach(k => delete store[k]);
  Object.assign(store, prestore || {});
  for (const k in els) delete els[k];
  // wrap so that the page's top-level bindings are reachable
  return new Function(script + '\nreturn {S, ITEMS, get i(){return i}, importLabels, importDraft, csvRows, cur};')();
}
const key = k => handlers.keydown({ key: k, preventDefault() {} });

const out = {};
// 1. suggestion pre-fills position; choosing a direction + Space checks it as accepted
let p = load();
out.first = p.ITEMS[p.i].id;
out.prefilled = p.cur().position || null;
out.sugline = els.sugline.textContent;
key(' ');                                   // no direction yet -> refused
out.blocked = !p.cur().checked;
key('w'); key(' ');
const firstId = p.ITEMS[0].id;
out.accepted = { checked: p.S[firstId].checked, sug_accepted: p.S[firstId].sug_accepted, pos: p.S[firstId].position };
// 2. second item: change the suggested position -> not accepted
const second = p.ITEMS[p.i].id;
key('4'); key('q'); key(' ');
out.changed = { checked: p.S[second].checked, sug_accepted: p.S[second].sug_accepted, pos: p.S[second].position };
// 3. export: unchecked rows are empty
out.rows = p.csvRows();
// 4. AI draft import: loaded unchecked; does not overwrite a reviewed, different entry
const draft = 'id,position,direction,crosswalk,occluded,boundary,excluded,mask_error,note\n' +
  p.ITEMS.map((t, j) => `${t.id},3,e,0,0,0,0,0,""`).join('\n');
out.draftLoaded = p.importDraft(draft);
out.afterDraft = { firstKept: p.S[firstId].position, firstChecked: p.S[firstId].checked,
                   thirdDraft: p.S[p.ITEMS[2].id].draft, thirdChecked: p.S[p.ITEMS[2].id].checked };
// 5. labels saved by the old page (no checked flag) count as checked
const old = {}; old[p.ITEMS[2].id] = { position: '2', direction: 'q' };
p = load({ [KEY]: JSON.stringify(old) });
out.legacyChecked = p.S[p.ITEMS[2].id].checked === true;
// 6. toggling suggestions off clears an unreviewed suggestion
p = load();
const before = p.cur().position;
key('a');
out.toggleOff = { before, after: p.cur().position || null, stored: store[KEY + '_sug'] };
console.log(JSON.stringify(out));
