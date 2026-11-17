declare const __APP_VERSION__: string;

export const APP_VERSION: string =
  typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '0.1.1';

const GITHUB_LATEST_RELEASE_API =
  'https://api.github.com/repos/NDCLI/music-studio-ai/releases/latest';
const GITHUB_RELEASES_PAGE = 'https://github.com/NDCLI/music-studio-ai/releases';

export interface AppUpdateCheckResult {
  currentVersion: string;
  latestVersion: string;
  hasUpdate: boolean;
  releaseUrl: string;
  installerUrl?: string;
  installerName?: string;
  publishedAt?: string;
  releaseNotes?: string;
}

/**
 * Compares two semantic version strings (e.g., "0.1.2" vs "v0.1.1").
 * Returns > 0 if `a > b`, 0 if equal, < 0 if `a < b`.
 */
export function compareSemver(a: string, b: string): number {
  const parse = (v: string) =>
    v
      .trim()
      .replace(/^v/i, '')
      .split('-')[0]
      .split('.')
      .map((n) => parseInt(n, 10) || 0);
  const pa = parse(a);
  const pb = parse(b);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const diff = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

/**
 * Fetches latest release metadata from GitHub Releases and checks if a newer version exists.
 */
export async function checkForAppUpdates(): Promise<AppUpdateCheckResult> {
  const res = await fetch(GITHUB_LATEST_RELEASE_API, {
    headers: {
      Accept: 'application/vnd.github+json',
    },
  });

  if (res.status === 404) {
    return {
      currentVersion: APP_VERSION,
      latestVersion: APP_VERSION,
      hasUpdate: false,
      releaseUrl: GITHUB_RELEASES_PAGE,
    };
  }

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} khi kết nối GitHub Releases`);
  }

  const data = (await res.json()) as {
    tag_name?: string;
    html_url?: string;
    published_at?: string;
    body?: string;
    assets?: { name?: string; browser_download_url?: string }[];
  };

  const latestRaw = (data.tag_name ?? APP_VERSION).replace(/^v/i, '');
  const hasUpdate = compareSemver(latestRaw, APP_VERSION) > 0;

  const exeAsset = data.assets?.find((a) =>
    a.name?.toLowerCase().endsWith('-setup.exe')
  );

  return {
    currentVersion: APP_VERSION,
    latestVersion: latestRaw,
    hasUpdate,
    releaseUrl: data.html_url ?? GITHUB_RELEASES_PAGE,
    installerUrl: exeAsset?.browser_download_url,
    installerName: exeAsset?.name,
    publishedAt: data.published_at,
    releaseNotes: data.body?.trim(),
  };
}
