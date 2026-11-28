// Label_Sheets (race spec, Requirement 9; design D12, D13).
//
// Classic script (no ES modules), so print/labels.html works when opened from
// the file system. Lays out the Tilemark labels at exact size: each Label
// element is 114 x 149 mm in CSS millimetres, two per A4 sheet, turned
// 90 degrees. Images come from labels/<Product_ID>.png; a missing image is
// drawn as an outline of the same size with the Product_ID.
//
// Exposes globalThis.TilemarkLabels for tests. Runs init() only when the page
// has loaded products.js (globalThis.TILEMARK_PRODUCTS).
;(function (root) {
  'use strict'

  var LABEL_W_MM = 114
  var LABEL_H_MM = 149
  var DEFAULT_COPIES = 6
  var MAX_COPIES = 60
  var PX_PER_MM = 7.68 // label.png resolution, 195.072 dpi
  var SLOT_LEFT_MM = 30.5 // (210 - 149) / 2
  var SLOT_TOPS_MM = [12, 136]
  var CAPTION_OFFSET_MM = 115 // caption top, from the slot top (slot is 114 mm high)
  var RULER_MM = 100
  var TOLERANCE_MM = 0.5
  var SVG_NS = 'http://www.w3.org/2000/svg'

  /** Copies input value -> integer in [0, 60]; anything not a number gives the default 6. */
  function clampCopies(value) {
    var n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN
    if (!isFinite(n)) return DEFAULT_COPIES
    return Math.min(MAX_COPIES, Math.max(0, Math.round(n)))
  }

  /** [{ id, copies }] -> sheets, each an array of 1 or 2 { id, copy, of }, in product and copy order. */
  function planSheets(entries) {
    var labels = []
    entries.forEach(function (e) {
      for (var c = 1; c <= e.copies; c++) labels.push({ id: e.id, copy: c, of: e.copies })
    })
    var sheets = []
    for (var i = 0; i < labels.length; i += 2) sheets.push(labels.slice(i, i + 2))
    return sheets
  }

  /** The CSS size of every Label element, in millimetres, never pixels. */
  function labelSize() {
    return { width: LABEL_W_MM + 'mm', height: LABEL_H_MM + 'mm' }
  }

  function el(doc, tag, className, text) {
    var node = doc.createElement(tag)
    if (className) node.className = className
    if (text !== undefined) node.textContent = text
    return node
  }

  function svgEl(doc, tag, attrs) {
    var node = doc.createElementNS(SVG_NS, tag)
    Object.keys(attrs).forEach(function (k) {
      node.setAttribute(k, String(attrs[k]))
    })
    return node
  }

  /** 100 mm Ruler_Line with a tick every 10 mm. */
  function ruler(doc) {
    var svg = svgEl(doc, 'svg', {
      class: 'ruler',
      width: RULER_MM + 'mm',
      height: '8mm',
      viewBox: '0 0 ' + RULER_MM + ' 8',
      'data-mm': RULER_MM,
    })
    svg.appendChild(svgEl(doc, 'line', { x1: 0, y1: 0.5, x2: RULER_MM, y2: 0.5, stroke: '#000', 'stroke-width': 0.25 }))
    for (var x = 0; x <= RULER_MM; x += 10) {
      var long = x % 50 === 0
      svg.appendChild(svgEl(doc, 'line', { x1: x, y1: 0.5, x2: x, y2: long ? 4 : 2.5, stroke: '#000', 'stroke-width': 0.25 }))
    }
    var t = svgEl(doc, 'text', { x: RULER_MM / 2, y: 7.5, 'font-size': 3, 'text-anchor': 'middle' })
    t.textContent = '100 mm'
    svg.appendChild(t)
    return svg
  }

  function labelElement(doc, item, available) {
    var size = labelSize()
    var node
    if (available.has(item.id)) {
      node = el(doc, 'img', 'label')
      node.src = 'labels/' + item.id + '.png'
      node.alt = 'Tilemark label ' + item.id
    } else {
      node = el(doc, 'div', 'label outline')
      node.appendChild(el(doc, 'div', 'outline-id', String(item.id)))
      node.appendChild(el(doc, 'div', 'outline-note', 'labels/' + item.id + '.png not found'))
    }
    node.setAttribute('data-product-id', String(item.id))
    node.style.width = size.width
    node.style.height = size.height
    return node
  }

  /** Builds the Sheets into container. available: Set of Product_IDs whose image loaded. */
  function renderSheets(doc, container, plan, available) {
    while (container.firstChild) container.removeChild(container.firstChild)
    plan.forEach(function (labels, s) {
      var sheet = el(doc, 'div', 'sheet')
      sheet.setAttribute('data-sheet', String(s + 1))
      labels.forEach(function (item, i) {
        var slot = el(doc, 'div', 'slot')
        slot.style.left = SLOT_LEFT_MM + 'mm'
        slot.style.top = SLOT_TOPS_MM[i] + 'mm'
        slot.appendChild(labelElement(doc, item, available))
        sheet.appendChild(slot)
        var caption = el(doc, 'div', 'caption', 'ID ' + item.id + ' \u00b7 copy ' + item.copy + ' of ' + item.of)
        caption.style.top = SLOT_TOPS_MM[i] + CAPTION_OFFSET_MM + 'mm'
        sheet.appendChild(caption)
      })
      var rulerBox = el(doc, 'div', 'ruler-box')
      rulerBox.appendChild(ruler(doc))
      sheet.appendChild(rulerBox)
      sheet.appendChild(el(doc, 'div', 'actual-size', 'print at 100% (actual size)'))
      sheet.appendChild(el(doc, 'div', 'sheet-no', 'sheet ' + (s + 1) + ' of ' + plan.length))
      container.appendChild(sheet)
    })
  }

  /**
   * On-screen size check (9.10): every Label element's layout size (offsetWidth and
   * offsetHeight, before the 90 degree turn) against the 100 mm Ruler_Line.
   */
  function checkSizes(container) {
    var rulerNode = container.querySelector('.ruler')
    var labels = container.querySelectorAll('.label')
    if (!rulerNode) return { ok: labels.length === 0, count: labels.length, bad: [], pxPerMm: null }
    var pxPerMm = rulerNode.getBoundingClientRect().width / RULER_MM
    var bad = []
    for (var i = 0; i < labels.length; i++) {
      var w = labels[i].offsetWidth / pxPerMm
      var h = labels[i].offsetHeight / pxPerMm
      if (!(Math.abs(w - LABEL_W_MM) <= TOLERANCE_MM && Math.abs(h - LABEL_H_MM) <= TOLERANCE_MM)) {
        bad.push({ id: labels[i].getAttribute('data-product-id'), widthMm: w, heightMm: h })
      }
    }
    return { ok: bad.length === 0, count: labels.length, bad: bad, pxPerMm: pxPerMm }
  }

  function probeImage(doc, id) {
    return new Promise(function (resolve) {
      var img = doc.createElement('img')
      img.onload = function () {
        resolve({ id: id, ok: true, width: img.naturalWidth, height: img.naturalHeight })
      }
      img.onerror = function () {
        resolve({ id: id, ok: false })
      }
      img.src = 'labels/' + id + '.png'
    })
  }

  function init(doc, products) {
    var controls = doc.getElementById('copies')
    var container = doc.getElementById('sheets')
    var checkLine = doc.getElementById('check')
    var imageLine = doc.getElementById('images')
    var inputs = {}
    var available = new Set()

    products.forEach(function (p) {
      var label = el(doc, 'label', 'copies-input')
      label.appendChild(doc.createTextNode(p.name + ' (' + p.id + ') '))
      var input = el(doc, 'input')
      input.type = 'number'
      input.min = '0'
      input.max = String(MAX_COPIES)
      input.step = '1'
      input.value = String(DEFAULT_COPIES)
      input.addEventListener('change', update)
      inputs[p.id] = input
      label.appendChild(input)
      controls.appendChild(label)
    })

    function update() {
      var entries = products.map(function (p) {
        var copies = clampCopies(inputs[p.id].value)
        inputs[p.id].value = String(copies)
        return { id: p.id, copies: copies }
      })
      var plan = planSheets(entries)
      renderSheets(doc, container, plan, available)
      var r = checkSizes(container)
      var total = entries.reduce(function (a, e) {
        return a + e.copies
      }, 0)
      var head = total + ' labels on ' + plan.length + ' sheets. '
      if (r.ok) {
        checkLine.className = 'ok'
        checkLine.textContent =
          head + 'Size check: all ' + r.count + ' labels measure 114 \u00d7 149 mm' +
          (r.pxPerMm ? ' (1 mm = ' + r.pxPerMm.toFixed(3) + ' screen px).' : '.')
      } else {
        checkLine.className = 'bad'
        checkLine.textContent =
          head + 'Size check FAILED for ' + r.bad.length + ' of ' + r.count + ' labels: ' +
          r.bad.map(function (b) { return b.id + ' ' + b.widthMm.toFixed(2) + ' \u00d7 ' + b.heightMm.toFixed(2) + ' mm' }).join(', ')
      }
      checkLine.setAttribute('data-ok', String(r.ok))
    }

    Promise.all(products.map(function (p) { return probeImage(doc, p.id) })).then(function (probes) {
      imageLine.textContent = probes
        .map(function (r) {
          if (!r.ok) return 'labels/' + r.id + '.png not found: outline drawn.'
          available.add(r.id)
          var ppm = r.width / LABEL_W_MM
          var warn = Math.abs(ppm - PX_PER_MM) / PX_PER_MM > 0.01 ? ' (expected 7.68: check the file)' : ''
          return 'labels/' + r.id + '.png: ' + r.width + ' \u00d7 ' + r.height + ' px, ' + ppm.toFixed(2) + ' px per mm' + warn + '.'
        })
        .join(' ')
      update()
    })
  }

  root.TilemarkLabels = {
    LABEL_W_MM: LABEL_W_MM,
    LABEL_H_MM: LABEL_H_MM,
    DEFAULT_COPIES: DEFAULT_COPIES,
    MAX_COPIES: MAX_COPIES,
    SLOT_LEFT_MM: SLOT_LEFT_MM,
    SLOT_TOPS_MM: SLOT_TOPS_MM,
    clampCopies: clampCopies,
    planSheets: planSheets,
    labelSize: labelSize,
    renderSheets: renderSheets,
    checkSizes: checkSizes,
    init: init,
  }

  if (root.document && root.TILEMARK_PRODUCTS) init(root.document, root.TILEMARK_PRODUCTS)
})(globalThis)
