/* ════════════════════════════════════════════════════════════
   UNVEIL Code Browser — UI logic
   - Fetches manifest.json, renders the file tree.
   - Routes selection via location.hash: #file=path or #dir=path.
   - Loads file content from ./src/<urlencoded path>, renders as
     either highlight.js code or marked markdown.
   - For dir routes, renders a GitHub-style listing of children.
═══════════════════════════════════════════════════════════════ */

(() => {
  'use strict';

  const MANIFEST_URL = './manifest.json';
  const SRC_PREFIX = './src/';
  const STORAGE_KEY = 'unveil-code-browser-folders';

  const state = {
    manifest: null,
    fileIndex: new Map(),   // path → file entry
    dirIndex: new Map(),    // path → dir entry
    current: null,          // {type: 'file'|'dir', path}
    openFolders: new Set(),
  };

  // ── DOM refs ──────────────────────────────────────────────
  const $tree = document.getElementById('cb-tree');
  const $breadcrumb = document.getElementById('cb-breadcrumb');
  const $fileName = document.getElementById('cb-file-name');
  const $langBadge = document.getElementById('cb-lang-badge');
  const $fileSize = document.getElementById('cb-file-size');
  const $emptyState = document.getElementById('cb-empty-state');
  const $markdown = document.getElementById('cb-markdown');
  const $codeWrap = document.getElementById('cb-code-wrap');
  const $code = document.getElementById('cb-code');
  const $lineNumbers = document.getElementById('cb-line-numbers');
  const $dirView = document.getElementById('cb-dir-view');
  const $dirList = document.getElementById('cb-dir-list');
  const $copyBtn = document.getElementById('cb-copy-btn');
  const $rawBtn = document.getElementById('cb-raw-btn');
  const $collapseAll = document.getElementById('cb-collapse-all');
  const $downloadBtn = document.getElementById('cb-download-btn');

  // ── Persistence for folder expansion state ───────────────
  const loadOpenFolders = () => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored) {
        state.openFolders = new Set(JSON.parse(stored));
      }
    } catch (e) { /* ignore */ }
  };
  const saveOpenFolders = () => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...state.openFolders]));
    } catch (e) { /* ignore */ }
  };

  // ── Helpers ──────────────────────────────────────────────
  const encodePath = (p) => p.split('/').map(encodeURIComponent).join('/');

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const indexEntries = (entries) => {
    for (const e of entries) {
      if (e.type === 'file') {
        state.fileIndex.set(e.path, e);
      } else if (e.type === 'dir') {
        state.dirIndex.set(e.path, e);
        indexEntries(e.children);
      }
    }
  };

  const getParentFolders = (filePath) => {
    const parts = filePath.split('/').slice(0, -1);
    const folders = [];
    for (let i = 0; i < parts.length; i++) {
      folders.push(parts.slice(0, i + 1).join('/'));
    }
    return folders;
  };

  // ── Tree rendering ───────────────────────────────────────
  const svgFolder = `<svg viewBox="0 0 16 16" fill="currentColor" width="14" height="14" aria-hidden="true"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3h-6.5a.25.25 0 0 1-.2-.1L6.06 1.4A1.75 1.75 0 0 0 4.66 1z"/></svg>`;
  const svgFile = `<svg viewBox="0 0 16 16" fill="currentColor" width="13" height="13" aria-hidden="true"><path d="M2 1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 12.25 16h-8.5A1.75 1.75 0 0 1 2 14.25zm10.5 5.379V14.25a.25.25 0 0 1-.25.25h-8.5a.25.25 0 0 1-.25-.25V1.75a.25.25 0 0 1 .25-.25H8V4.75c0 .967.784 1.75 1.75 1.75z"/></svg>`;
  const svgChevron = `<svg viewBox="0 0 16 16" fill="currentColor" width="10" height="10" aria-hidden="true"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 1 1-1.06-1.06L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06z"/></svg>`;

  const renderNode = (entry, depth) => {
    if (entry.type === 'dir') {
      const wrapper = document.createElement('div');
      wrapper.className = 'cb-dir-wrapper';

      const isOpen = state.openFolders.has(entry.path);

      const node = document.createElement('div');
      node.className = 'cb-node cb-dir' + (isOpen ? ' cb-open' : '');
      node.style.paddingLeft = `${0.5 + depth * 0.9}rem`;
      node.dataset.path = entry.path;
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
      for (const child of entry.children) {
        childContainer.appendChild(renderNode(child, depth + 1));
      }

      node.addEventListener('click', () => {
        const willOpen = !node.classList.contains('cb-open');
        node.classList.toggle('cb-open', willOpen);
        childContainer.classList.toggle('cb-open', willOpen);
        node.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
        if (willOpen) {
          state.openFolders.add(entry.path);
        } else {
          state.openFolders.delete(entry.path);
        }
        saveOpenFolders();
        // Only navigate when expanding — collapsing should just collapse the
        // sidebar entry without re-routing (and without applyHash re-opening it).
        if (willOpen) {
          navigateTo('dir', entry.path);
        }
      });

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

  const renderTree = () => {
    $tree.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (const entry of state.manifest.tree) {
      frag.appendChild(renderNode(entry, 0));
    }
    $tree.appendChild(frag);
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

  const expandAncestors = (filePath) => {
    const folders = getParentFolders(filePath);
    for (const folder of folders) {
      state.openFolders.add(folder);
      const node = $tree.querySelector(`.cb-node.cb-dir[data-path="${CSS.escape(folder)}"]`);
      if (node) {
        node.classList.add('cb-open');
        node.setAttribute('aria-expanded', 'true');
        const children = node.parentElement.querySelector(':scope > .cb-children');
        if (children) children.classList.add('cb-open');
      }
    }
    saveOpenFolders();
  };

  // ── Breadcrumb ───────────────────────────────────────────
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

  // ── File loading & rendering ─────────────────────────────
  const hideAllViews = () => {
    $emptyState.hidden = true;
    $markdown.hidden = true;
    $codeWrap.hidden = true;
    $dirView.hidden = true;
  };

  const showEmpty = () => {
    hideAllViews();
    $emptyState.hidden = false;
    $fileName.textContent = '—';
    $langBadge.textContent = '';
    $fileSize.textContent = '';
  };

  const showMarkdown = (text, entry) => {
    hideAllViews();
    $markdown.hidden = false;
    const rendered = window.marked
      ? window.marked.parse(text, { mangle: false, headerIds: false })
      : `<pre>${escapeHtml(text)}</pre>`;
    $markdown.innerHTML = rendered;
    // Syntax-highlight any fenced code blocks inside the markdown.
    if (window.hljs) {
      $markdown.querySelectorAll('pre code').forEach(block => {
        window.hljs.highlightElement(block);
      });
    }
    $fileName.textContent = entry.path;
    $langBadge.textContent = entry.lang || '';
    $fileSize.textContent = formatSize(entry.size);
  };

  const showCode = (text, entry) => {
    hideAllViews();
    $codeWrap.hidden = false;

    $code.className = '';
    if (entry.lang && entry.lang !== 'plaintext') {
      $code.classList.add(`language-${entry.lang}`);
    }
    $code.textContent = text;

    const lineCount = text.length === 0 ? 1 : (text.match(/\n/g) || []).length + 1;
    const nums = new Array(lineCount);
    for (let i = 0; i < lineCount; i++) nums[i] = (i + 1).toString();
    $lineNumbers.textContent = nums.join('\n');

    if (window.hljs && entry.lang && entry.lang !== 'plaintext') {
      try { window.hljs.highlightElement($code); }
      catch (e) { /* fall back to plain */ }
    }

    $fileName.textContent = entry.path;
    $langBadge.textContent = entry.lang || '';
    $fileSize.textContent = formatSize(entry.size);
  };

  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const loadFile = async (entry) => {
    const url = SRC_PREFIX + encodePath(entry.path);
    $rawBtn.onclick = () => window.open(url, '_blank', 'noopener');
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      if (entry.lang === 'markdown') {
        showMarkdown(text, entry);
      } else {
        showCode(text, entry);
      }
    } catch (err) {
      hideAllViews();
      $codeWrap.hidden = false;
      $code.className = '';
      $code.textContent = `Failed to load ${entry.path}\n\n${err.message}`;
      $lineNumbers.textContent = '1\n2\n3';
    }
  };

  // ── Directory view ───────────────────────────────────────
  const svgFolderLarge = `<svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16" aria-hidden="true"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3h-6.5a.25.25 0 0 1-.2-.1L6.06 1.4A1.75 1.75 0 0 0 4.66 1z"/></svg>`;
  const svgFileLarge = `<svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16" aria-hidden="true"><path d="M2 1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 12.25 16h-8.5A1.75 1.75 0 0 1 2 14.25zm10.5 5.379V14.25a.25.25 0 0 1-.25.25h-8.5a.25.25 0 0 1-.25-.25V1.75a.25.25 0 0 1 .25-.25H8V4.75c0 .967.784 1.75 1.75 1.75z"/></svg>`;
  const svgUp = `<svg viewBox="0 0 16 16" fill="currentColor" width="16" height="16" aria-hidden="true"><path d="M8.53 1.22a.75.75 0 0 0-1.06 0L3.22 5.47a.75.75 0 0 0 1.06 1.06l2.97-2.97v9.69a.75.75 0 0 0 1.5 0V3.56l2.97 2.97a.75.75 0 1 0 1.06-1.06z"/></svg>`;

  const loadDir = (entry) => {
    hideAllViews();
    $dirView.hidden = false;
    $fileName.textContent = entry.path + '/';
    $langBadge.textContent = 'directory';
    $fileSize.textContent = `${entry.children.length} item${entry.children.length === 1 ? '' : 's'}`;

    $dirList.innerHTML = '';

    // Parent "up" row, except at top-level paths
    const parentPath = entry.path.includes('/')
      ? entry.path.split('/').slice(0, -1).join('/')
      : null;
    if (parentPath !== null) {
      const upRow = document.createElement('div');
      upRow.className = 'cb-dir-row cb-dir-row-up';
      upRow.setAttribute('role', 'listitem');
      upRow.innerHTML =
        `<span class="cb-dir-row-icon">${svgUp}</span>` +
        `<span class="cb-dir-row-name">..</span>`;
      upRow.addEventListener('click', () => navigateTo('dir', parentPath));
      $dirList.appendChild(upRow);
    }

    // Folders first, then files — alphabetised within each group
    const sorted = [...entry.children].sort((a, b) => {
      if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    if (sorted.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'cb-dir-empty';
      empty.textContent = 'This folder is empty.';
      $dirList.appendChild(empty);
      return;
    }

    for (const child of sorted) {
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

    // Reset viewer-body scroll
    document.getElementById('cb-viewer-body').scrollTop = 0;
  };

  // ── Routing ──────────────────────────────────────────────
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

  const applyHash = () => {
    const parsed = parseHash() || { type: 'file', path: state.manifest.default_file };
    state.current = parsed;

    if (parsed.type === 'dir') {
      const entry = state.dirIndex.get(parsed.path);
      if (!entry) {
        showEmpty();
        updateBreadcrumb('');
        updateTreeSelection(null, null);
        return;
      }
      // expandAncestors walks parents of the given path; append "/_" so the
      // dir itself is treated as a parent and gets opened too.
      expandAncestors(parsed.path + '/_');
      updateTreeSelection('dir', parsed.path);
      updateBreadcrumb(parsed.path);
      loadDir(entry);
      return;
    }

    // file
    const entry = state.fileIndex.get(parsed.path);
    if (!entry) {
      showEmpty();
      updateBreadcrumb('');
      updateTreeSelection(null, null);
      return;
    }
    expandAncestors(parsed.path);
    updateTreeSelection('file', parsed.path);
    updateBreadcrumb(parsed.path);
    loadFile(entry);
  };

  // ── Copy button ──────────────────────────────────────────
  const wireCopyButton = () => {
    $copyBtn.addEventListener('click', async () => {
      let text = '';
      if (!$codeWrap.hidden) {
        text = $code.textContent || '';
      } else if (!$markdown.hidden) {
        // Copy raw markdown by re-fetching (cheap; already cached).
        if (state.current && state.current.type === 'file') {
          try {
            const r = await fetch(SRC_PREFIX + encodePath(state.current.path));
            text = await r.text();
          } catch (e) { /* ignore */ }
        }
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
      } catch (e) { /* clipboard blocked */ }
    });
  };

  // ── On-demand zip download ───────────────────────────────
  // The zip is built in the browser from the live ./src/ files so it always
  // matches what the visitor sees. No static Submission.zip on disk.
  const collectFiles = (entries, acc) => {
    for (const e of entries) {
      if (e.type === 'file') acc.push(e);
      else if (e.type === 'dir') collectFiles(e.children, acc);
    }
    return acc;
  };

  const wireDownloadButton = () => {
    $downloadBtn.addEventListener('click', async () => {
      if (!window.JSZip) {
        alert('JSZip failed to load — cannot build the archive.');
        return;
      }
      const label = $downloadBtn.querySelector('span');
      const original = label ? label.textContent : '';
      $downloadBtn.disabled = true;
      if (label) label.textContent = 'Building zip…';

      try {
        const files = collectFiles(state.manifest.tree, []);
        const zip = new JSZip();
        // Match the conventional "Submission/<path>" layout reviewers expect.
        const root = (state.manifest.root || 'Submission') + '/';

        // Fetch every file in parallel, then add to the zip.
        const fetched = await Promise.all(files.map(async (f) => {
          const res = await fetch(SRC_PREFIX + encodePath(f.path));
          if (!res.ok) throw new Error(`${f.path}: HTTP ${res.status}`);
          return { path: f.path, blob: await res.blob() };
        }));
        for (const { path, blob } of fetched) {
          zip.file(root + path, blob);
        }

        const archive = await zip.generateAsync({
          type: 'blob',
          compression: 'DEFLATE',
          compressionOptions: { level: 6 },
        });
        const url = URL.createObjectURL(archive);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'UNVEIL-codes.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1500);
      } catch (err) {
        alert(`Could not build the zip: ${err.message}`);
      } finally {
        $downloadBtn.disabled = false;
        if (label) label.textContent = original;
      }
    });
  };

  // ── Collapse-all ────────────────────────────────────────
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

  // ── Boot ─────────────────────────────────────────────────
  const boot = async () => {
    loadOpenFolders();
    try {
      const res = await fetch(MANIFEST_URL);
      if (!res.ok) throw new Error(`manifest HTTP ${res.status}`);
      state.manifest = await res.json();
    } catch (err) {
      $tree.innerHTML = `<div class="cb-tree-loading">Failed to load manifest: ${escapeHtml(err.message)}</div>`;
      return;
    }
    indexEntries(state.manifest.tree);
    renderTree();
    wireCopyButton();
    wireCollapseAll();
    wireDownloadButton();
    window.addEventListener('hashchange', applyHash);
    applyHash();
  };

  boot();
})();
