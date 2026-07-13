/* ════════════════════════════════════════════════════════════
   UNVEIL Anonymized G1 — Cloudflare Worker reverse proxy.

   Hides the upstream dataset URL from browser DevTools by exposing two
   neutral routes and fetching from the real host server-side:

     GET  /tree[/<path-in-repo>]   → upstream tree-listing API
     GET  /file/<path-in-repo>     → upstream raw-file resolve URL
                                     (the Worker follows the 307 redirect
                                      internally so the browser never sees
                                      the CDN URL either)

   Paste this into a fresh Cloudflare Worker; no environment secrets are
   required while the upstream dataset is public.

   Local reference copy lives at anonymized/cloudflare-worker.js — edit
   here, then redeploy via the CF dashboard.
═══════════════════════════════════════════════════════════════ */

const REPO     = 'sihatafnan/dummy_g1';
const REV      = 'main';
const HF       = 'https://huggingface.co';
const TREE_UP  = `${HF}/api/datasets/${REPO}/tree/${REV}`;
const FILE_UP  = `${HF}/datasets/${REPO}/resolve/${REV}`;

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

    // Tree route: /tree or /tree/<sub-path>
    if (p === '/tree' || p === '/tree/' || p.startsWith('/tree/')) {
      const sub = p === '/tree' || p === '/tree/' ? '' : p.slice('/tree'.length);
      const upstream = `${TREE_UP}${sub}${url.search}`;
      return relay(upstream, /*follow=*/ false, 'application/json');
    }

    // File route: /file/<path-in-repo>
    if (p.startsWith('/file/')) {
      const filePath = p.slice('/file/'.length);
      if (!filePath) {
        return new Response('Missing path', { status: 400, headers: CORS });
      }
      const upstream = `${FILE_UP}/${filePath}`;
      return relay(upstream, /*follow=*/ true, 'text/csv; charset=utf-8');
    }

    // Friendly root help so accidental visitors aren't met with a 404 wall.
    if (p === '/' || p === '') {
      return new Response(
        'OK. Use /tree[/<path>] for listings or /file/<path> for raw files.',
        { status: 200, headers: { ...CORS, 'Content-Type': 'text/plain' } },
      );
    }

    return new Response('Not Found', { status: 404, headers: CORS });
  },
};

async function relay(upstream, follow, fallbackContentType) {
  let r;
  try {
    r = await fetch(upstream, {
      redirect: follow ? 'follow' : 'manual',
      // Forward only the headers we need; Worker -> HF must look like a
      // plain anonymous client so nothing about the visitor leaks upstream.
      headers: { 'User-Agent': 'g1-anon-proxy/1.0' },
    });
  } catch (err) {
    return new Response('Upstream fetch failed: ' + (err && err.message),
      { status: 502, headers: { ...CORS, 'Content-Type': 'text/plain' } });
  }

  const headers = {
    ...CORS,
    'Content-Type':  r.headers.get('Content-Type')  || fallbackContentType,
    'Cache-Control': r.headers.get('Cache-Control') || 'public, max-age=300',
  };
  // Preserve content length when available so the browser can show progress.
  const len = r.headers.get('Content-Length');
  if (len) headers['Content-Length'] = len;

  return new Response(r.body, { status: r.status, headers });
}
