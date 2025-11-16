'use strict';

// Generic PME GUI utilities: param specs, GUI generation, param I/O.

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
    var row = document.createElement('div');
    row.className = 'row';

    var label = document.createElement('label');
    label.htmlFor = s.inputId;
    label.textContent = s.label || s.uniform.substring(1);
    row.appendChild(label);

    var input = document.createElement('input');
    input.id = s.inputId;
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
    var el = document.getElementById(s.inputId);
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
    params[s.uniform] = v;
  }
  return params;
}

function applyParamsToUniformsWithSpecs(paramSpecs, uniforms, params) {
  for (var i = 0; i < paramSpecs.length; i++) {
    var s = paramSpecs[i];
    if (uniforms[s.uniform]) {
      uniforms[s.uniform].value = params[s.uniform];
    }
  }
}

function buildStatusStringFromParams(paramSpecs, params, Mode, siteIdx) {
  var parts = [];
  for (var i = 0; i < paramSpecs.length; i++) {
    var s = paramSpecs[i];
    var v = params[s.uniform];
    var name = s.uniform.substring(1);
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

if (typeof window !== 'undefined') {
  window.createPmeGui                   = createPmeGui;
  window.readParamsFromInputsWithSpecs  = readParamsFromInputsWithSpecs;
  window.applyParamsToUniformsWithSpecs = applyParamsToUniformsWithSpecs;
  window.buildStatusStringFromParams    = buildStatusStringFromParams;
}
