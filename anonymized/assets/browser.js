/* ════════════════════════════════════════════════════════════
   UNVEIL Anonymized-Trajectories Browser
   - Lazy tree fetched from a remote dataset host.
   - File content fetched from the same host on demand.
   - Routing via location.hash:  #file=path  or  #dir=path
═══════════════════════════════════════════════════════════════ */

(() => {
  'use strict';

  // ── Configuration ──────────────────────────────────────────
  // All upstream dataset access goes through a neutral reverse proxy so the
  // browser's network tab never reveals the real host or repo identifier.
  const PROXY_BASE    = 'https://g1-anon.unveil2000.workers.dev';
  const TREE_API_BASE = `${PROXY_BASE}/tree`;
  const RESOLVE_BASE  = `${PROXY_BASE}/file`;

  const PAGE_LIMIT = 1000;              // HF API max per page
  const ROOT_PATH = '';                 // sentinel for repo root
  const FILE_MAX_BYTES = 50 * 1024 * 1024;  // 50 MB safety cap when fetching file content
  const STORAGE_KEY = 'unveil-anon-browser-folders';

  // ── State ──────────────────────────────────────────────────
  // treeCache: dirPath ('' for root) -> array of HF entries normalized to
  //   { type: 'file'|'dir', name, path, size? }
  // pendingFetches: dirPath -> Promise (de-dup concurrent loads)
  const state = {
    treeCache: new Map(),
    pendingFetches: new Map(),
    current: null,
    openFolders: new Set(),
    rootListed: false,
  };

  // ── DOM refs ───────────────────────────────────────────────
  const $tree         = document.getElementById('cb-tree');
  const $breadcrumb   = document.getElementById('cb-breadcrumb');
  const $fileName     = document.getElementById('cb-file-name');
  const $langBadge    = document.getElementById('cb-lang-badge');
  const $fileSize     = document.getElementById('cb-file-size');
  const $emptyState   = document.getElementById('cb-empty-state');
  const $codeWrap     = document.getElementById('cb-code-wrap');
  const $code         = document.getElementById('cb-code');
  const $lineNumbers  = document.getElementById('cb-line-numbers');
  const $markdown     = document.getElementById('cb-markdown');
  const $dirView      = document.getElementById('cb-dir-view');
  const $dirList      = document.getElementById('cb-dir-list');
  const $copyBtn      = document.getElementById('cb-copy-btn');
  const $rawBtn       = document.getElementById('cb-raw-btn');
  const $collapseAll  = document.getElementById('cb-collapse-all');
  const $viewerBody   = document.getElementById('cb-viewer-body');
  const $search       = document.getElementById('cb-search');
  const $searchClear  = document.getElementById('cb-search-clear');
  const $searchStatus = document.getElementById('cb-search-status');
  const $searchResults = document.getElementById('cb-search-results');
  const $csvView      = document.getElementById('cb-csv-view');
  const $csvThead     = document.getElementById('cb-csv-thead');
  const $csvTbody     = document.getElementById('cb-csv-tbody');
  const $viewToggle   = document.getElementById('cb-view-toggle');
  const $viewToggleLabel = document.getElementById('cb-view-toggle-label');

  // Manifest of all CSV paths (loaded once, lazily). Used by the search box
  // for client-side fuzzy matching without per-keystroke API hits.
  let manifestEntries = null;        // array of full paths "csv/<date>/<file>"
  let manifestLoading = null;        // Promise during initial fetch
  const SEARCH_MAX_RESULTS = 200;    // cap shown to keep DOM cheap

  // CSV viewer state — currently-loaded file's parsed text + view mode
  const csvState = {
    text: null,           // raw CSV text of the file currently shown
    path: null,           // the path of that file
    size: null,           // hint
    mode: 'table',        // 'table' | 'raw'
  };
  const CSV_MAX_ROWS = 5000;   // cap rendered rows to keep the DOM responsive
  const CSV_FLOAT_DIGITS = 4;  // decimals shown for float cells

  // ── Persistence ────────────────────────────────────────────
  const loadOpenFolders = () => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored) state.openFolders = new Set(JSON.parse(stored));
    } catch (_) { /* ignore */ }
  };
  const saveOpenFolders = () => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...state.openFolders]));
    } catch (_) { /* ignore */ }
  };

  // ── Helpers ────────────────────────────────────────────────
  const encodePath = (p) => p.split('/').map(encodeURIComponent).join('/');

  const formatSize = (bytes) => {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const basename = (p) => {
    const i = p.lastIndexOf('/');
    return i < 0 ? p : p.slice(i + 1);
  };

  const dirname = (p) => {
    const i = p.lastIndexOf('/');
    return i < 0 ? '' : p.slice(0, i);
  };

  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Parse RFC-5988 Link header: <url>; rel="next", <url>; rel="prev"
  const parseNextLink = (header) => {
    if (!header) return null;
    const m = header.match(/<([^>]+)>\s*;\s*rel="next"/);
    return m ? m[1] : null;
  };

  // ── HF tree fetch (paginated, cached, de-duped) ────────────
  const fetchTreeEntries = (dirPath) => {
    if (state.treeCache.has(dirPath)) {
      return Promise.resolve(state.treeCache.get(dirPath));
    }
    if (state.pendingFetches.has(dirPath)) {
      return state.pendingFetches.get(dirPath);
    }
    const initialUrl =
      TREE_API_BASE + (dirPath ? '/' + encodePath(dirPath) : '') + `?limit=${PAGE_LIMIT}`;
    const promise = (async () => {
      const entries = [];
      let url = initialUrl;
      while (url) {
        const res = await fetch(url);
        if (!res.ok) {
          if (res.status === 401 || res.status === 403) {
            throw new Error(`Access denied (${res.status}) for ${dirPath || '(root)'}.`);
          }
          if (res.status === 404) {
            throw new Error(`Path not found: ${dirPath || '(root)'}`);
          }
          throw new Error(`HTTP ${res.status} for ${dirPath || '(root)'}`);
        }
        const page = await res.json();
        for (const e of page) {
          // HF returns: { type: 'file'|'directory', path, size, oid, ... }
          const isDir = e.type === 'directory';
          entries.push({
            type: isDir ? 'dir' : 'file',
            name: basename(e.path),
            path: e.path,
            size: isDir ? undefined : (e.size ?? 0),
          });
        }
        url = parseNextLink(res.headers.get('Link'));
      }
      // Folders first, then files; alphabetised within each group
      entries.sort((a, b) => {
        if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
        return a.name.localeCompare(b.name, undefined, { numeric: true });
      });
      state.treeCache.set(dirPath, entries);
      return entries;
    })();
    state.pendingFetches.set(dirPath, promise);
    promise.finally(() => state.pendingFetches.delete(dirPath));
    return promise;
  };

  // ── Tree rendering ─────────────────────────────────────────
  const svgFolder = `<svg viewBox="0 0 16 16" fill="currentColor" width="14" height="14" aria-hidden="true"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3h-6.5a.25.25 0 0 1-.2-.1L6.06 1.4A1.75 1.75 0 0 0 4.66 1z"/></svg>`;
  const svgFile = `<svg viewBox="0 0 16 16" fill="currentColor" width="13" height="13" aria-hidden="true"><path d="M2 1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 12.25 16h-8.5A1.75 1.75 0 0 1 2 14.25zm10.5 5.379V14.25a.25.25 0 0 1-.25.25h-8.5a.25.25 0 0 1-.25-.25V1.75a.25.25 0 0 1 .25-.25H8V4.75c0 .967.784 1.75 1.75 1.75z"/></svg>`;
  const svgChevron = `<svg viewBox="0 0 16 16" fill="currentColor" width="10" height="10" aria-hidden="true"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 1 1-1.06-1.06L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06z"/></svg>`;
  const svgSpinner = `<svg viewBox="0 0 16 16" fill="currentColor" width="10" height="10" aria-hidden="true" class="cb-spin"><path d="M8 1.5a6.5 6.5 0 0 0-6.5 6.5h1.5a5 5 0 0 1 5-5z"/></svg>`;

  // Build a node DOM element for an entry. Folders carry a child container
  // that's populated lazily on first expand.
  const renderNode = (entry, depth) => {
    if (entry.type === 'dir') {
      const wrapper = document.createElement('div');
      wrapper.className = 'cb-dir-wrapper';

      const isOpen = state.openFolders.has(entry.path);

      const node = document.createElement('div');
      node.className = 'cb-node cb-dir' + (isOpen ? ' cb-open' : '');
      node.style.paddingLeft = `${0.5 + depth * 0.9}rem`;
      node.dataset.path = entry.path;
      node.dataset.loaded = '0';
      node.setAttribute('role', 'treeitem');
      node.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      node.innerHTML =
        `<span class="cb-node-chevron">${svgChevron}</span>` +
        `<span class="cb-node-icon">${svgFolder}</span>` +
        `<span class="cb-node-name"></span>`;
      node.querySelector('.cb-node-name').textContent = entry.name;

      const childContainer = document.createElement('div');
      childContainer.className = 'cb-children' + (isOpen ? ' cb-open' : '');
      childContainer.setAttribute('role', 'group');
      childContainer.dataset.depth = String(depth + 1);

      const ensureChildren = async () => {
        if (node.dataset.loaded === '1') return;
        node.dataset.loaded = '1';
        // Loading placeholder
        childContainer.innerHTML = '';
        const loading = document.createElement('div');
        loading.className = 'cb-tree-loading';
        loading.style.paddingLeft = `${0.5 + (depth + 1) * 0.9}rem`;
        loading.textContent = 'Loading…';
        childContainer.appendChild(loading);
        try {
          const children = await fetchTreeEntries(entry.path);
          childContainer.innerHTML = '';
          for (const child of children) {
            childContainer.appendChild(renderNode(child, depth + 1));
          }
          if (children.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'cb-tree-loading';
            empty.style.paddingLeft = `${0.5 + (depth + 1) * 0.9}rem`;
            empty.textContent = '(empty)';
            childContainer.appendChild(empty);
          }
        } catch (err) {
          node.dataset.loaded = '0';  // allow retry
          childContainer.innerHTML = '';
          const fail = document.createElement('div');
          fail.className = 'cb-tree-loading';
          fail.style.paddingLeft = `${0.5 + (depth + 1) * 0.9}rem`;
          fail.style.color = '#cf222e';
          fail.textContent = 'Failed to load: ' + err.message;
          childContainer.appendChild(fail);
        }
      };

      node.addEventListener('click', () => {
        const willOpen = !node.classList.contains('cb-open');
        node.classList.toggle('cb-open', willOpen);
        childContainer.classList.toggle('cb-open', willOpen);
        node.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
        if (willOpen) {
          state.openFolders.add(entry.path);
          ensureChildren();
        } else {
          state.openFolders.delete(entry.path);
        }
        saveOpenFolders();
        // Only navigate (= update viewer pane) on expand
        if (willOpen) navigateTo('dir', entry.path);
      });

      // If we restored this folder as open, eager-load its children now
      if (isOpen) ensureChildren();

      wrapper.appendChild(node);
      wrapper.appendChild(childContainer);
      return wrapper;
    }

    // File node
    const node = document.createElement('div');
    node.className = 'cb-node cb-file';
    node.style.paddingLeft = `${0.5 + depth * 0.9 + 0.85}rem`;
    node.dataset.path = entry.path;
    node.setAttribute('role', 'treeitem');
    node.innerHTML =
      `<span class="cb-node-icon">${svgFile}</span>` +
      `<span class="cb-node-name"></span>`;
    node.querySelector('.cb-node-name').textContent = entry.name;
    node.addEventListener('click', () => navigateTo('file', entry.path));
    return node;
  };

  const renderRootTree = async () => {
    $tree.innerHTML = '<div class="cb-tree-loading">Loading file tree…</div>';
    try {
      const entries = await fetchTreeEntries(ROOT_PATH);
      $tree.innerHTML = '';
      const frag = document.createDocumentFragment();
      for (const entry of entries) frag.appendChild(renderNode(entry, 0));
      $tree.appendChild(frag);
      state.rootListed = true;
      if (entries.length === 0) {
        $tree.innerHTML = '<div class="cb-tree-loading">(repo is empty — upload still in progress?)</div>';
      }
    } catch (err) {
      $tree.innerHTML =
        `<div class="cb-tree-loading" style="color:#cf222e">Failed to load tree: ${escapeHtml(err.message)}</div>`;
    }
  };

  const updateTreeSelection = (type, path) => {
    $tree.querySelectorAll('.cb-node.cb-selected').forEach(n => {
      n.classList.remove('cb-selected');
    });
    if (!path) return;
    const selector = type === 'dir'
      ? `.cb-node.cb-dir[data-path="${CSS.escape(path)}"]`
      : `.cb-node.cb-file[data-path="${CSS.escape(path)}"]`;
    const node = $tree.querySelector(selector);
    if (node) {
      node.classList.add('cb-selected');
      node.scrollIntoView({ block: 'nearest' });
    }
  };

  // Walk ancestor dirs, fetching+expanding each so the target row is visible.
  const expandAncestors = async (targetPath) => {
    const parts = targetPath.split('/');
    // parents = ['csv', 'csv/210531', ...] (each prefix except the leaf itself)
    const parents = [];
    for (let i = 1; i <= parts.length - 1; i++) {
      parents.push(parts.slice(0, i).join('/'));
    }
    for (const folder of parents) {
      state.openFolders.add(folder);
      // Ensure the dir is rendered in the sidebar; if not, we need to fetch
      // its parent first to reveal it. Walk top-down via fetchTreeEntries +
      // DOM expansion.
      const parentOfFolder = dirname(folder);
      if (!state.treeCache.has(parentOfFolder)) {
        try { await fetchTreeEntries(parentOfFolder); } catch (_) { return; }
      }
      const node = $tree.querySelector(`.cb-node.cb-dir[data-path="${CSS.escape(folder)}"]`);
      if (node) {
        node.classList.add('cb-open');
        node.setAttribute('aria-expanded', 'true');
        const childWrap = node.parentElement.querySelector(':scope > .cb-children');
        if (childWrap) childWrap.classList.add('cb-open');
        // Trigger lazy load if needed
        if (node.dataset.loaded === '0') {
          // Click would toggle off+on; manually load instead
          node.dataset.loaded = '1';
          try {
            const children = await fetchTreeEntries(folder);
            if (childWrap) {
              childWrap.innerHTML = '';
              for (const child of children) {
                childWrap.appendChild(renderNode(child, parseInt(childWrap.dataset.depth || '1', 10)));
              }
            }
          } catch (_) { node.dataset.loaded = '0'; }
        }
      }
    }
    saveOpenFolders();
  };

  // ── Breadcrumb ─────────────────────────────────────────────
  const updateBreadcrumb = (path) => {
    $breadcrumb.innerHTML = '';
    if (!path) return;
    const parts = path.split('/');
    for (let i = 0; i < parts.length; i++) {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'cb-crumb-sep';
        sep.textContent = '/';
        $breadcrumb.appendChild(sep);
      }
      const isLast = i === parts.length - 1;
      const crumb = document.createElement('span');
      crumb.className = 'cb-crumb' + (isLast ? ' cb-crumb-active' : ' cb-crumb-link');
      crumb.textContent = parts[i];
      if (!isLast) {
        const ancestorPath = parts.slice(0, i + 1).join('/');
        crumb.addEventListener('click', () => navigateTo('dir', ancestorPath));
      }
      $breadcrumb.appendChild(crumb);
    }
  };

  // ── View toggles ───────────────────────────────────────────
  const hideAllViews = () => {
    $emptyState.hidden = true;
    $codeWrap.hidden = true;
    $dirView.hidden = true;
    if ($markdown) $markdown.hidden = true;
    if ($csvView)  $csvView.hidden  = true;
    if ($viewToggle) $viewToggle.hidden = true;   // hidden by default; CSV path shows it
  };

  const showEmpty = (msg) => {
    hideAllViews();
    $emptyState.hidden = false;
    if (msg) {
      const p = $emptyState.querySelector('p');
      if (p) p.textContent = msg;
    }
    $fileName.textContent = '—';
    $langBadge.textContent = '';
    $fileSize.textContent = '';
  };

  const showMarkdown = (text, path, size) => {
    hideAllViews();
    $markdown.hidden = false;
    const rendered = window.marked
      ? window.marked.parse(text, { mangle: false, headerIds: false })
      : `<pre>${escapeHtml(text)}</pre>`;
    $markdown.innerHTML = rendered;
    if (window.hljs) {
      $markdown.querySelectorAll('pre code').forEach(block => {
        try { window.hljs.highlightElement(block); } catch (_) { /* ignore */ }
      });
    }
    $fileName.textContent = path;
    $langBadge.textContent = 'markdown';
    $fileSize.textContent = formatSize(size);
    $viewerBody.scrollTop = 0;
  };

  const showCode = (text, path, size) => {
    hideAllViews();
    $codeWrap.hidden = false;

    $code.className = '';
    // CSVs aren't a default highlight.js language; render as plaintext.
    $code.textContent = text;

    const lineCount = text.length === 0 ? 1 : (text.match(/\n/g) || []).length + 1;
    const nums = new Array(lineCount);
    for (let i = 0; i < lineCount; i++) nums[i] = (i + 1).toString();
    $lineNumbers.textContent = nums.join('\n');

    $fileName.textContent = path;
    const ext = (path.split('.').pop() || '').toLowerCase();
    $langBadge.textContent = ext;
    $fileSize.textContent = formatSize(size);
    $viewerBody.scrollTop = 0;
  };

  // ── CSV table renderer ─────────────────────────────────────
  // Format a single cell value: integers as-is, floats trimmed to a fixed
  // number of decimals (trailing zeros stripped), non-numeric kept literal.
  const formatCsvCell = (v) => {
    if (v === null || v === undefined || v === '') return '';
    if (typeof v === 'number') {
      if (Number.isInteger(v)) return String(v);
      return v.toFixed(CSV_FLOAT_DIGITS).replace(/\.?0+$/, '');
    }
    return String(v);
  };

  // Build the table DOM from parsed rows. First row is the header.
  const renderCsvTable = (rows) => {
    $csvThead.innerHTML = '';
    $csvTbody.innerHTML = '';
    if (!rows || rows.length === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.className = 'cb-csv-empty';
      td.colSpan = 1;
      td.textContent = '(empty CSV)';
      tr.appendChild(td);
      $csvTbody.appendChild(tr);
      return;
    }

    const header = rows[0];
    const headRow = document.createElement('tr');
    for (const h of header) {
      const th = document.createElement('th');
      th.textContent = String(h);
      headRow.appendChild(th);
    }
    $csvThead.appendChild(headRow);

    const dataRows = rows.slice(1);
    const cap = Math.min(dataRows.length, CSV_MAX_ROWS);
    const frag = document.createDocumentFragment();
    for (let i = 0; i < cap; i++) {
      const r = dataRows[i];
      const tr = document.createElement('tr');
      for (let c = 0; c < header.length; c++) {
        const td = document.createElement('td');
        td.textContent = formatCsvCell(r[c]);
        tr.appendChild(td);
      }
      frag.appendChild(tr);
    }
    $csvTbody.appendChild(frag);

    if (dataRows.length > CSV_MAX_ROWS) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = header.length;
      td.className = 'cb-csv-truncated';
      td.textContent = `Showing first ${CSV_MAX_ROWS.toLocaleString()} of ${dataRows.length.toLocaleString()} rows — toggle to Raw view for the full file.`;
      tr.appendChild(td);
      $csvTbody.appendChild(tr);
    }
  };

  const showCsv = (text, path, size) => {
    csvState.text = text;
    csvState.path = path;
    csvState.size = size;
    csvState.mode = 'table';

    hideAllViews();
    $csvView.hidden = false;
    $viewToggle.hidden = false;
    $viewToggleLabel.textContent = 'Raw';
    $viewToggle.title = 'Switch to raw text view';

    $fileName.textContent = path;
    $langBadge.textContent = 'csv';
    $fileSize.textContent = formatSize(size);

    // PapaParse may not have loaded yet (CDN failure / slow). Fall back to raw.
    if (!window.Papa) {
      console.warn('PapaParse not loaded; falling back to raw view.');
      csvState.mode = 'raw';
      showCode(text, path, size);
      return;
    }

    // Parse with dynamic typing so numeric cells become Numbers (better
    // formatting). skipEmptyLines avoids stray trailing-newline rows.
    const parsed = window.Papa.parse(text, {
      header: false,
      dynamicTyping: true,
      skipEmptyLines: true,
    });
    renderCsvTable(parsed.data);
    $viewerBody.scrollTop = 0;
    $viewerBody.scrollLeft = 0;
  };

  // Toggle between table and raw for the currently-loaded CSV.
  const toggleCsvView = () => {
    if (!csvState.text) return;
    if (csvState.mode === 'table') {
      csvState.mode = 'raw';
      showCode(csvState.text, csvState.path, csvState.size);
      $viewToggle.hidden = false;
      $viewToggleLabel.textContent = 'Table';
      $viewToggle.title = 'Switch back to table view';
    } else {
      csvState.mode = 'table';
      showCsv(csvState.text, csvState.path, csvState.size);
    }
  };

  // ── File loading ───────────────────────────────────────────
  const loadFile = async (path, sizeHint) => {
    const url = RESOLVE_BASE + '/' + encodePath(path);
    $rawBtn.onclick = () => window.open(url, '_blank', 'noopener');
    if (sizeHint != null && sizeHint > FILE_MAX_BYTES) {
      hideAllViews();
      $codeWrap.hidden = false;
      $code.className = '';
      $code.textContent = `File is ${formatSize(sizeHint)} — too large to render inline.\nUse the Raw button to download it.`;
      $lineNumbers.textContent = '1\n2';
      $fileName.textContent = path;
      $langBadge.textContent = 'too large';
      $fileSize.textContent = formatSize(sizeHint);
      return;
    }
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const ext = (path.split('.').pop() || '').toLowerCase();
      const size = sizeHint ?? text.length;
      if (ext === 'md' || ext === 'markdown') {
        showMarkdown(text, path, size);
      } else if (ext === 'csv') {
        showCsv(text, path, size);
      } else {
        // Non-CSV file: clear any stale CSV state so the view toggle is hidden.
        csvState.text = null;
        showCode(text, path, size);
      }
    } catch (err) {
      hideAllViews();
      $codeWrap.hidden = false;
      $code.className = '';
      $code.textContent = `Failed to load ${path}\n\n${err.message}\n\nThe file may not yet be available — try again later.`;
      $lineNumbers.textContent = '1\n2\n3\n4\n5';
      $fileName.textContent = path;
      $langBadge.textContent = 'error';
      $fileSize.textContent = '';
    }
  };

  // ── Directory listing ──────────────────────────────────────
  const svgFolderLarge = `<svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16" aria-hidden="true"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3h-6.5a.25.25 0 0 1-.2-.1L6.06 1.4A1.75 1.75 0 0 0 4.66 1z"/></svg>`;
  const svgFileLarge = `<svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16" aria-hidden="true"><path d="M2 1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 12.25 16h-8.5A1.75 1.75 0 0 1 2 14.25zm10.5 5.379V14.25a.25.25 0 0 1-.25.25h-8.5a.25.25 0 0 1-.25-.25V1.75a.25.25 0 0 1 .25-.25H8V4.75c0 .967.784 1.75 1.75 1.75z"/></svg>`;
  const svgUp = `<svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16" aria-hidden="true"><path d="M8.53 1.22a.75.75 0 0 0-1.06 0L3.22 5.47a.75.75 0 0 0 1.06 1.06l2.97-2.97v9.69a.75.75 0 0 0 1.5 0V3.56l2.97 2.97a.75.75 0 1 0 1.06-1.06z"/></svg>`;

  const renderDirListing = (dirPath, entries) => {
    hideAllViews();
    $dirView.hidden = false;
    $fileName.textContent = (dirPath || '/') + (dirPath ? '/' : '');
    $langBadge.textContent = 'directory';
    $fileSize.textContent = `${entries.length} item${entries.length === 1 ? '' : 's'}`;

    $dirList.innerHTML = '';

    // Parent ".." row, unless at repo root
    if (dirPath !== '') {
      const parentPath = dirname(dirPath);
      const upRow = document.createElement('div');
      upRow.className = 'cb-dir-row cb-dir-row-up';
      upRow.setAttribute('role', 'listitem');
      upRow.innerHTML =
        `<span class="cb-dir-row-icon">${svgUp}</span>` +
        `<span class="cb-dir-row-name">..</span>`;
      upRow.addEventListener('click', () => navigateTo('dir', parentPath));
      $dirList.appendChild(upRow);
    }

    if (entries.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'cb-dir-empty';
      empty.textContent = 'This folder is empty (or files have not finished uploading yet).';
      $dirList.appendChild(empty);
      $viewerBody.scrollTop = 0;
      return;
    }

    for (const child of entries) {
      const row = document.createElement('div');
      row.className = 'cb-dir-row ' + (child.type === 'dir' ? 'cb-dir-row-folder' : 'cb-dir-row-file');
      row.setAttribute('role', 'listitem');
      const iconClass = child.type === 'dir' ? 'cb-dir-row-icon cb-dir-icon-folder' : 'cb-dir-row-icon';
      const icon = child.type === 'dir' ? svgFolderLarge : svgFileLarge;
      const sizeStr = child.type === 'file' ? formatSize(child.size) : '';
      row.innerHTML =
        `<span class="${iconClass}">${icon}</span>` +
        `<span class="cb-dir-row-name"></span>` +
        `<span class="cb-dir-row-size">${sizeStr}</span>`;
      row.querySelector('.cb-dir-row-name').textContent = child.name;
      row.addEventListener('click', () => navigateTo(child.type, child.path));
      $dirList.appendChild(row);
    }

    $viewerBody.scrollTop = 0;
  };

  const loadDir = async (dirPath) => {
    hideAllViews();
    $dirView.hidden = false;
    $fileName.textContent = (dirPath || '/') + (dirPath ? '/' : '');
    $langBadge.textContent = 'directory';
    $fileSize.textContent = 'loading…';
    $dirList.innerHTML = '<div class="cb-dir-empty">Loading…</div>';
    try {
      const entries = await fetchTreeEntries(dirPath);
      renderDirListing(dirPath, entries);
    } catch (err) {
      $dirList.innerHTML = '';
      const fail = document.createElement('div');
      fail.className = 'cb-dir-empty';
      fail.style.color = '#cf222e';
      fail.textContent = 'Failed to load: ' + err.message;
      $dirList.appendChild(fail);
      $fileSize.textContent = '';
    }
  };

  // ── Routing ────────────────────────────────────────────────
  const parseHash = () => {
    const h = location.hash.replace(/^#/, '');
    if (!h) return null;
    let m = h.match(/(?:^|&)file=([^&]+)/);
    if (m) return { type: 'file', path: decodeURIComponent(m[1]) };
    m = h.match(/(?:^|&)dir=([^&]+)/);
    if (m) return { type: 'dir', path: decodeURIComponent(m[1]) };
    return null;
  };

  const navigateTo = (type, path) => {
    const key = type === 'dir' ? 'dir' : 'file';
    const target = `#${key}=${encodeURIComponent(path).replace(/%2F/g, '/')}`;
    if (location.hash !== target) {
      location.hash = target;
    } else {
      applyHash();
    }
  };

  // Look up a file entry's size via its parent dir's cached listing (best effort)
  const findFileEntry = (path) => {
    const parent = dirname(path);
    const cached = state.treeCache.get(parent);
    if (!cached) return null;
    return cached.find(e => e.type === 'file' && e.path === path) || null;
  };

  const applyHash = async () => {
    const parsed = parseHash();
    if (!parsed) {
      showEmpty();
      updateBreadcrumb('');
      updateTreeSelection(null, null);
      return;
    }
    state.current = parsed;

    if (parsed.type === 'dir') {
      await expandAncestors(parsed.path + '/_');
      updateTreeSelection('dir', parsed.path);
      updateBreadcrumb(parsed.path);
      loadDir(parsed.path);
      return;
    }

    // file
    await expandAncestors(parsed.path);
    updateTreeSelection('file', parsed.path);
    updateBreadcrumb(parsed.path);
    const entry = findFileEntry(parsed.path);
    loadFile(parsed.path, entry ? entry.size : null);
  };

  // ── Copy ───────────────────────────────────────────────────
  const wireCopyButton = () => {
    $copyBtn.addEventListener('click', async () => {
      // CSV table view → copy the cached raw text. Code view → copy what's
      // rendered. Anything else (markdown/dir/empty) → nothing to copy.
      let text = '';
      if ($csvView && !$csvView.hidden && csvState.text) {
        text = csvState.text;
      } else if (!$codeWrap.hidden) {
        text = $code.textContent || '';
      }
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        const label = $copyBtn.querySelector('.cb-action-label');
        const original = label ? label.textContent : '';
        $copyBtn.classList.add('cb-action-success');
        if (label) label.textContent = 'Copied';
        setTimeout(() => {
          $copyBtn.classList.remove('cb-action-success');
          if (label) label.textContent = original;
        }, 1400);
      } catch (_) { /* clipboard blocked */ }
    });
  };

  // ── Collapse-all ───────────────────────────────────────────
  const wireCollapseAll = () => {
    $collapseAll.addEventListener('click', () => {
      state.openFolders.clear();
      saveOpenFolders();
      $tree.querySelectorAll('.cb-node.cb-dir.cb-open').forEach(n => {
        n.classList.remove('cb-open');
        n.setAttribute('aria-expanded', 'false');
      });
      $tree.querySelectorAll('.cb-children.cb-open').forEach(c => {
        c.classList.remove('cb-open');
      });
    });
  };

  // ── Search ─────────────────────────────────────────────────
  const loadManifest = () => {
    if (manifestEntries) return Promise.resolve(manifestEntries);
    if (manifestLoading) return manifestLoading;
    manifestLoading = (async () => {
      const res = await fetch('./manifest.json');
      if (!res.ok) throw new Error(`manifest HTTP ${res.status}`);
      const data = await res.json();
      // Flatten {date: [file, …]} → ["csv/<date>/<file>", …]
      const flat = [];
      for (const date of Object.keys(data)) {
        for (const file of data[date]) flat.push(`csv/${date}/${file}`);
      }
      manifestEntries = flat;
      return flat;
    })();
    return manifestLoading;
  };

  // Token-based substring matcher. Query is lowercased, split on whitespace
  // and underscores; a path matches when EVERY token appears as a substring
  // of the lowercased path. Returns the first SEARCH_MAX_RESULTS matches.
  const runSearch = (rawQuery) => {
    if (!manifestEntries) return null;     // not ready yet
    const q = rawQuery.toLowerCase().trim();
    if (!q) return [];
    // Split on whitespace ONLY so underscores in the query (e.g. "jumping_rop_R")
    // stay literal — matches the exact path substring, not each underscore-piece.
    const tokens = q.split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return [];
    const out = [];
    for (const path of manifestEntries) {
      const lp = path.toLowerCase();
      let ok = true;
      for (const t of tokens) {
        if (!lp.includes(t)) { ok = false; break; }
      }
      if (ok) {
        out.push(path);
        if (out.length >= SEARCH_MAX_RESULTS) break;
      }
    }
    return out;
  };

  // Escape special regex chars in user input so we can build a highlighter.
  const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // Highlight each query token within `text` (case-insensitive).
  const highlightHtml = (text, tokens) => {
    if (tokens.length === 0) return escapeHtml(text);
    const pattern = new RegExp('(' + tokens.map(escapeRegex).join('|') + ')', 'gi');
    return escapeHtml(text).replace(pattern, '<mark>$1</mark>');
  };

  const renderSearchResults = (query, results) => {
    $searchResults.innerHTML = '';
    if (results === null) {
      $searchStatus.textContent = 'Loading file index…';
      return;
    }
    if (results.length === 0) {
      $searchStatus.textContent = `No matches for “${query}”`;
      return;
    }
    const tokens = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    const cap = results.length >= SEARCH_MAX_RESULTS;
    $searchStatus.textContent = cap
      ? `Showing first ${SEARCH_MAX_RESULTS} matches — refine your search to narrow.`
      : `${results.length} match${results.length === 1 ? '' : 'es'}`;

    const frag = document.createDocumentFragment();
    for (const path of results) {
      const file = basename(path);
      const dir  = dirname(path);
      const row = document.createElement('div');
      row.className = 'cb-search-result';
      row.setAttribute('role', 'option');
      row.dataset.path = path;
      row.innerHTML =
        `<span class="cb-search-result-name">${highlightHtml(file, tokens)}</span>` +
        `<span class="cb-search-result-dir">${highlightHtml(dir, tokens)}</span>`;
      row.addEventListener('click', () => navigateTo('file', path));
      frag.appendChild(row);
    }
    $searchResults.appendChild(frag);
  };

  // Show search panel (results + status), hide the file tree.
  const showSearchPanel = () => {
    $tree.hidden = true;
    $searchResults.hidden = false;
    $searchStatus.hidden = false;
    $searchClear.hidden = false;
  };
  // Hide search panel, show the file tree again.
  const hideSearchPanel = () => {
    $tree.hidden = false;
    $searchResults.hidden = true;
    $searchStatus.hidden = true;
    $searchClear.hidden = true;
    $searchResults.innerHTML = '';
    $searchStatus.textContent = '';
  };

  const wireSearch = () => {
    let debounceTimer = null;
    const handleInput = () => {
      const q = $search.value;
      if (!q.trim()) { hideSearchPanel(); return; }
      showSearchPanel();

      // Kick off manifest load on first non-empty input
      if (!manifestEntries) {
        renderSearchResults(q, null);
        loadManifest().then(() => {
          // Re-run with the latest query value (user may have typed more)
          renderSearchResults($search.value, runSearch($search.value));
        }).catch(err => {
          $searchStatus.textContent = 'Search index failed to load: ' + err.message;
        });
        return;
      }
      renderSearchResults(q, runSearch(q));
    };

    $search.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(handleInput, 80);
    });
    // Esc clears
    $search.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        $search.value = '';
        hideSearchPanel();
        $search.blur();
      }
    });
    $searchClear.addEventListener('click', () => {
      $search.value = '';
      hideSearchPanel();
      $search.focus();
    });
  };

  const wireViewToggle = () => {
    $viewToggle.addEventListener('click', toggleCsvView);
  };

  // ── Boot ───────────────────────────────────────────────────
  const boot = async () => {
    loadOpenFolders();
    wireCopyButton();
    wireCollapseAll();
    wireSearch();
    wireViewToggle();
    window.addEventListener('hashchange', applyHash);
    await renderRootTree();
    // Apply initial route once tree is ready (so expandAncestors finds nodes)
    await applyHash();
    // Pre-warm the manifest in the background so the first search is instant.
    loadManifest().catch(() => { /* surfaced when user actually searches */ });
  };

  boot();
})();
