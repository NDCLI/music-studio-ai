import { LocalAiModelSpec } from '../types/studio';

const DB_NAME = 'MusicStudioAiOnnxCache';
const STORE_NAME = 'onnx_models';
const DB_VERSION = 1;

export interface CachedModelRecord {
  id: string;
  fileName: string;
  sizeBytes: number;
  sha256Hex: string;
  source: 'huggingface' | 'local_import';
  updatedAtIso: string;
  blob: Blob;
}

function openCacheDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Không thể mở IndexedDB'));
  });
}

/**
 * Computes lowercase SHA-256 hex digest using WebCrypto subtle API.
 */
export async function computeBufferSha256(buffer: ArrayBuffer): Promise<string> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const bytes = new Uint8Array(hashBuffer);
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

/**
 * Validates that an ArrayBuffer represents a genuine binary ONNX protobuf (not an HTML 404/LFS stub).
 */
export function validateOnnxBinaryHeader(bytes: Uint8Array): string | null {
  if (bytes.byteLength < 16) {
    return 'File quá nhỏ để là một model ONNX hợp lệ.';
  }
  const headStr = new TextDecoder().decode(bytes.slice(0, 32)).toLowerCase();
  if (
    headStr.startsWith('<!doctype') ||
    headStr.startsWith('<html') ||
    headStr.startsWith('version https://')
  ) {
    return 'File tải về không phải trọng số ONNX (có thể là trang HTML lỗi hoặc con trỏ Git LFS).';
  }
  return null;
}

export async function listCachedModelsFromDb(): Promise<
  Record<string, Omit<CachedModelRecord, 'blob'>>
> {
  const db = await openCacheDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();
    req.onsuccess = () => {
      const map: Record<string, Omit<CachedModelRecord, 'blob'>> = {};
      for (const rec of (req.result as CachedModelRecord[]) ?? []) {
        map[rec.id] = {
          id: rec.id,
          fileName: rec.fileName,
          sizeBytes: rec.sizeBytes,
          sha256Hex: rec.sha256Hex,
          source: rec.source,
          updatedAtIso: rec.updatedAtIso,
        };
      }
      resolve(map);
    };
    req.onerror = () => reject(req.error);
  });
}

async function saveCachedModelRecord(record: CachedModelRecord): Promise<void> {
  const db = await openCacheDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function deleteCachedModelFromDb(modelId: string): Promise<void> {
  const db = await openCacheDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).delete(modelId);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Imports a user-selected local `.onnx` file, computes SHA-256, validates binary header,
 * and caches it offline in IndexedDB.
 */
export async function importLocalOnnxModel(
  file: File,
  targetSpec: LocalAiModelSpec
): Promise<Omit<CachedModelRecord, 'blob'>> {
  if (!file.name.toLowerCase().endsWith('.onnx')) {
    throw new Error('Chỉ hỗ trợ nhập file định dạng .onnx.');
  }

  const arrayBuf = await file.arrayBuffer();
  const headerErr = validateOnnxBinaryHeader(new Uint8Array(arrayBuf));
  if (headerErr) {
    throw new Error(headerErr);
  }

  const sha256Hex = await computeBufferSha256(arrayBuf);
  const record: CachedModelRecord = {
    id: targetSpec.id,
    fileName: file.name,
    sizeBytes: arrayBuf.byteLength,
    sha256Hex,
    source: 'local_import',
    updatedAtIso: new Date().toISOString(),
    blob: new Blob([arrayBuf], { type: 'application/octet-stream' }),
  };

  await saveCachedModelRecord(record);
  return {
    id: record.id,
    fileName: record.fileName,
    sizeBytes: record.sizeBytes,
    sha256Hex: record.sha256Hex,
    source: record.source,
    updatedAtIso: record.updatedAtIso,
  };
}

/**
 * Downloads an `.onnx` model from HuggingFace with streaming progress, pause/abort support,
 * SHA-256 checksum computation, and IndexedDB persistence.
 */
export async function downloadModelFromHuggingFace(
  spec: LocalAiModelSpec,
  abortSignal: AbortSignal,
  onProgress: (downloadedBytes: number, totalBytes: number, percent: number) => void
): Promise<Omit<CachedModelRecord, 'blob'>> {
  const response = await fetch(spec.downloadUrl, {
    signal: abortSignal,
    mode: 'cors',
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} khi tải ${spec.fileName} từ HuggingFace.`);
  }

  const headerLen = response.headers.get('content-length');
  const expectedTotal = headerLen ? parseInt(headerLen, 10) : spec.sizeMb * 1024 * 1024;
  const reader = response.body?.getReader();

  if (!reader) {
    throw new Error('Trình duyệt không hỗ trợ streaming response body.');
  }

  const chunks: Uint8Array[] = [];
  let downloaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      downloaded += value.byteLength;
      const pct = expectedTotal > 0 ? Math.min(99, Math.round((downloaded / expectedTotal) * 100)) : 50;
      onProgress(downloaded, expectedTotal, pct);
    }
  }

  const combined = new Uint8Array(downloaded);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }

  const headerErr = validateOnnxBinaryHeader(combined);
  if (headerErr) {
    throw new Error(headerErr);
  }

  const sha256Hex = await computeBufferSha256(combined.buffer);
  const record: CachedModelRecord = {
    id: spec.id,
    fileName: spec.fileName,
    sizeBytes: downloaded,
    sha256Hex,
    source: 'huggingface',
    updatedAtIso: new Date().toISOString(),
    blob: new Blob([combined], { type: 'application/octet-stream' }),
  };

  await saveCachedModelRecord(record);
  onProgress(downloaded, downloaded, 100);

  return {
    id: record.id,
    fileName: record.fileName,
    sizeBytes: record.sizeBytes,
    sha256Hex: record.sha256Hex,
    source: record.source,
    updatedAtIso: record.updatedAtIso,
  };
}
