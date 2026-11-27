import "@testing-library/jest-dom";
import { beforeEach } from "vitest";

// R3F's <Canvas> measures itself via react-use-measure, which depends on
// ResizeObserver. jsdom doesn't ship one, so the Sky View dome inside the
// wizard throws on mount in component tests. Stub it — tests don't need
// real layout measurements.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (!globalThis.ResizeObserver) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).ResizeObserver = ResizeObserverStub;
}


// Node's localStorage global is unavailable in some jsdom/Node combinations.
// Use an isolated in-memory implementation for component tests.
const testStorage = new Map<string, string>();
const localStorageShim: Storage = {
  get length() {
    return testStorage.size;
  },
  clear() {
    testStorage.clear();
  },
  getItem(key: string) {
    return testStorage.get(String(key)) ?? null;
  },
  key(index: number) {
    return [...testStorage.keys()][index] ?? null;
  },
  removeItem(key: string) {
    testStorage.delete(String(key));
  },
  setItem(key: string, value: string) {
    testStorage.set(String(key), String(value));
  },
};

Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: localStorageShim,
});

beforeEach(() => localStorageShim.clear());
