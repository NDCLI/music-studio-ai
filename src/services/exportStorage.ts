import { BrowserAudioExportFormat } from './dspEngine';

const EXPORT_HANDLE_DB = 'music-studio-export-handles-v1';
const EXPORT_HANDLE_STORE = 'handles';
const EXPORT_HANDLE_KEY = 'export_directory';

export interface BrowserWritableFileStream {
  write: (data: Blob) => Promise<void>;
  close: () => Promise<void>;
}

export interface BrowserFileHandle {
  name: string;
  createWritable: () => Promise<BrowserWritableFileStream>;
}

export interface BrowserDirectoryHandle {
  name: string;
  getFileHandle: (
    name: string,
    options?: { create?: boolean }
  ) => Promise<BrowserFileHandle>;
  queryPermission?: (desc?: { mode?: 'read' | 'readwrite' }) => Promise<'granted' | 'denied' | 'prompt'>;
  requestPermission?: (desc?: { mode?: 'read' | 'readwrite' }) => Promise<'granted' | 'denied' | 'prompt'>;
}

export function supportsDirectoryPicker(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof (window as unknown as { showDirectoryPicker?: unknown }).showDirectoryPicker === 'function'
  );
}

export function supportsSaveFilePicker(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof (window as unknown as { showSaveFilePicker?: unknown }).showSaveFilePicker === 'function'
  );
}

export function isEmbeddedFrame(): boolean {
  try {
    return typeof window !== 'undefined' && window.self !== window.top;
  } catch {
    return true;
  }
}

function openExportDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(EXPORT_HANDLE_DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(EXPORT_HANDLE_STORE)) {
        db.createObjectStore(EXPORT_HANDLE_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB unavailable'));
  });
}

export async function saveBrowserExportDirHandle(
  handle: BrowserDirectoryHandle
): Promise<void> {
  const db = await openExportDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(EXPORT_HANDLE_STORE, 'readwrite');
    tx.objectStore(EXPORT_HANDLE_STORE).put(handle, EXPORT_HANDLE_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Failed to persist export folder'));
  });
  db.close();
}

export async function loadBrowserExportDirHandle(): Promise<BrowserDirectoryHandle | null> {
  try {
    const db = await openExportDb();
    const handle = await new Promise<BrowserDirectoryHandle | null>((resolve, reject) => {
      const tx = db.transaction(EXPORT_HANDLE_STORE, 'readonly');
      const req = tx.objectStore(EXPORT_HANDLE_STORE).get(EXPORT_HANDLE_KEY);
      req.onsuccess = () => resolve((req.result as BrowserDirectoryHandle) ?? null);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return handle;
  } catch {
    return null;
  }
}

export async function clearBrowserExportDirHandle(): Promise<void> {
  try {
    const db = await openExportDb();
    await new Promise<void>((resolve) => {
      const tx = db.transaction(EXPORT_HANDLE_STORE, 'readwrite');
      tx.objectStore(EXPORT_HANDLE_STORE).delete(EXPORT_HANDLE_KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
    db.close();
  } catch {
    // Ignore storage cleanup errors
  }
}

export async function ensureDirectoryWritePermission(
  handle: BrowserDirectoryHandle,
  promptUser: boolean
): Promise<boolean> {
  if (typeof handle.queryPermission === 'function') {
    const current = await handle.queryPermission({ mode: 'readwrite' });
    if (current === 'granted') return true;
  }
  if (promptUser && typeof handle.requestPermission === 'function') {
    const requested = await handle.requestPermission({ mode: 'readwrite' });
    return requested === 'granted';
  }
  return typeof handle.queryPermission !== 'function';
}

export async function pickBrowserDirectoryHandle(): Promise<BrowserDirectoryHandle> {
  const picker = (
    window as unknown as {
      showDirectoryPicker?: (opts?: {
        mode?: 'read' | 'readwrite';
        startIn?: string;
      }) => Promise<BrowserDirectoryHandle>;
    }
  ).showDirectoryPicker;

  if (typeof picker !== 'function') {
    throw new Error('UNSUPPORTED_DIRECTORY_PICKER');
  }

  const handle = await picker({ mode: 'readwrite', startIn: 'music' });
  await saveBrowserExportDirHandle(handle).catch(() => {});
  return handle;
}

export async function pickSaveFileHandle(
  suggestedName: string,
  format: BrowserAudioExportFormat
): Promise<BrowserFileHandle> {
  const picker = (
    window as unknown as {
      showSaveFilePicker?: (opts?: {
        suggestedName?: string;
        startIn?: string;
        types?: { description: string; accept: Record<string, string[]> }[];
      }) => Promise<BrowserFileHandle>;
    }
  ).showSaveFilePicker;

  if (typeof picker !== 'function') {
    throw new Error('UNSUPPORTED_SAVE_FILE_PICKER');
  }

  const isFlac = format === 'flac';
  return picker({
    suggestedName,
    startIn: 'desktop',
    types: [
      isFlac
        ? { description: 'FLAC Audio', accept: { 'audio/flac': ['.flac'] } }
        : { description: 'WAV Audio', accept: { 'audio/wav': ['.wav'] } },
    ],
  });
}
