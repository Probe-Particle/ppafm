'use strict';

// Generic PME GUI utilities: param specs, GUI generation, param I/O.
// paramSpecs entries use `key` (canonical param name). Uniform and input IDs
// are derived by convention to avoid boilerplate duplication.

function _pmeInputIdFromKey(key) {
  return 'inp' + key.charAt(0).toUpperCase() + key.slice(1);
}

function _pmeUniformFromKey(key) {
  return 'u' + key.charAt(0).toUpperCase() + key.slice(1);
}

function createPmeGui(containerId, titleText, paramSpecs) {
  var container = document.getElementById(containerId);
  if (!container) return;

  // Clear container
  while (container.firstChild) container.removeChild(container.firstChild);

  var titleRow = document.createElement('div');
  titleRow.className = 'row';
  var strong = document.createElement('strong');
  strong.textContent = titleText;
  titleRow.appendChild(strong);
  container.appendChild(titleRow);

  for (var i = 0; i < paramSpecs.length; i++) {
    var s = paramSpecs[i];
    var key = s.key;
    var inputId = s.inputId || _pmeInputIdFromKey(key);

    var row = document.createElement('div');
    row.className = 'row';

    var label = document.createElement('label');
    label.htmlFor = inputId;
    label.textContent = s.label || key;
    row.appendChild(label);

    var input = document.createElement('input');
    input.id = inputId;
    input.type = 'number';
    if (s.min !== undefined) input.min = String(s.min);
    if (s.max !== undefined) input.max = String(s.max);
    if (s.step !== undefined) input.step = String(s.step);
    else input.step = (s.type === 'int') ? '1' : '0.01';
    input.value = String(s.def);
    row.appendChild(input);

    container.appendChild(row);
  }
}

function readParamsFromInputsWithSpecs(paramSpecs) {
  var params = {};
  for (var i = 0; i < paramSpecs.length; i++) {
    var s  = paramSpecs[i];
    var key = s.key;
    var inputId = s.inputId || _pmeInputIdFromKey(key);
    var el = document.getElementById(inputId);
    var v;
    if (!el) {
      v = s.def;
    } else {
      v = parseFloat(el.value);
      if (!isFinite(v)) v = s.def;
    }
    if (s.type === 'int') {
      v = Math.round(v);
    }
    if (s.min !== undefined && v < s.min) v = s.min;
    if (s.max !== undefined && v > s.max) v = s.max;
    params[key] = v;
  }
  return params;
}

function applyParamsToUniformsWithSpecs(paramSpecs, uniforms, params) {
  for (var i = 0; i < paramSpecs.length; i++) {
    var s = paramSpecs[i];
    var key = s.key;
    var uniformName = s.uniform || _pmeUniformFromKey(key);
    if (uniforms[uniformName]) {
      uniforms[uniformName].value = params[key];
    }
  }
}

function buildStatusStringFromParams(paramSpecs, params, Mode, siteIdx) {
  var parts = [];
  for (var i = 0; i < paramSpecs.length; i++) {
    var s = paramSpecs[i];
    var key = s.key;
    var v = params[key];
    var name = key;
    if (s.type === 'int') {
      parts.push(name + '=' + v);
    } else {
      parts.push(name + '=' + v.toFixed(2));
    }
  }
  parts.push('Mode=' + Mode);
  parts.push('siteIdx=' + siteIdx);
  return parts.join(', ');
}

// Parse a multiline text of sites into an array of numeric rows [x,y,z,E].
// Each non-empty line is split on whitespace and the first 4 tokens are
// parsed as floats. Invalid lines are skipped. The result is an array of
// arrays with shape [n,4], up to maxSites rows.
function parseSitesArrayFromText(text, maxSites) {
  var lines = (text || '').split(/\r?\n/);
  var sites = [];
  for (var i = 0; i < lines.length && sites.length < maxSites; i++) {
    var parts = lines[i].trim().split(/\s+/);
    if (parts.length < 4) continue;
    var row = [];
    var ok  = true;
    for (var j = 0; j < 4; j++) {
      var v = parseFloat(parts[j]);
      if (!isFinite(v)) { ok = false; break; }
      row.push(v);
    }
    if (!ok) continue;
    sites.push(row);
  }
  return sites;
}

if (typeof window !== 'undefined') {
  window.createPmeGui                   = createPmeGui;
  window.readParamsFromInputsWithSpecs  = readParamsFromInputsWithSpecs;
  window.applyParamsToUniformsWithSpecs = applyParamsToUniformsWithSpecs;
  window.buildStatusStringFromParams    = buildStatusStringFromParams;
  window.parseSitesArrayFromText        = parseSitesArrayFromText;
}
