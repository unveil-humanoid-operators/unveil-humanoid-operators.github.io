/* ════════════════════════════════════════════════════════════
   UNVEIL SMPL bins — Cloudflare Worker reverse proxy.

   Hides the upstream HuggingFace URL (sihatafnan/unveil-smpl-bins) from
   browser DevTools by exposing a neutral /file/<path> route and fetching
   from HF server-side. The browser only ever sees the Worker hostname.

     GET  /file/<bin-name>          → upstream raw-file resolve URL
                                      (Worker follows the 307 redirect
                                       internally so the browser never sees
                                       the HF CDN URL either)
     HEAD /file/faces.bin           → used by the correlation-bar-plot page
                                      to probe SMPL availability
     GET  /                         → friendly health message

   Paste this into a fresh Cloudflare Worker; no environment secrets are
   required while the upstream dataset is public.

   Local reference copy lives at anonymized/cloudflare-worker-smpl.js —
   edit here, then redeploy via the CF dashboard.
═══════════════════════════════════════════════════════════════ */

const REPO    = 'sihatafnan/unveil-smpl-bins';
const REV     = 'main';
const HF      = 'https://huggingface.co';
const FILE_UP = `${HF}/datasets/${REPO}/resolve/${REV}`;

const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
  'Access-Control-Allow-Headers': '*',
  'Access-Control-Max-Age':       '86400',
};

export default {
  async fetch(request) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method Not Allowed', { status: 405, headers: CORS });
    }

    const url = new URL(request.url);
    const p   = url.pathname;

    // File route: /file/<bin-name>
    if (p.startsWith('/file/')) {
      const filePath = p.slice('/file/'.length);
      if (!filePath) {
        return new Response('Missing path', { status: 400, headers: CORS });
      }
      const upstream = `${FILE_UP}/${filePath}`;
      return relay(upstream, request.method);
    }

    // Friendly root help.
    if (p === '/' || p === '') {
      return new Response(
        'OK. Use /file/<bin-name> for SMPL bin downloads.',
        { status: 200, headers: { ...CORS, 'Content-Type': 'text/plain' } },
      );
    }

    return new Response('Not Found', { status: 404, headers: CORS });
  },
};

async function relay(upstream, method) {
  let r;
  try {
    r = await fetch(upstream, {
      method,
      redirect: 'follow',
      // Forward only the headers we need; Worker -> HF must look like a
      // plain anonymous client so nothing about the visitor leaks upstream.
      headers: { 'User-Agent': 'unveil-smpl-proxy/1.0' },
    });
  } catch (err) {
    return new Response('Upstream fetch failed: ' + (err && err.message),
      { status: 502, headers: { ...CORS, 'Content-Type': 'text/plain' } });
  }

  const headers = {
    ...CORS,
    'Content-Type':  r.headers.get('Content-Type')  || 'application/octet-stream',
    'Cache-Control': r.headers.get('Cache-Control') || 'public, max-age=3600',
  };
  const len = r.headers.get('Content-Length');
  if (len) headers['Content-Length'] = len;

  return new Response(r.body, { status: r.status, headers });
}
