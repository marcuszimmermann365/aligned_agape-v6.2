
import { test, expect } from '@playwright/test';

test.describe('J5 Badge rendering', () => {
  test('renders badge in page', async ({ page }) => {
    // Loads static webui page; adjust if you serve via Flask
    await page.goto('file://' + process.cwd() + '/webui/index.html');
    const badge = page.locator('.badge');
    await expect(badge).toBeVisible();
    await expect(badge).toContainText(/J5/);
  });
});
