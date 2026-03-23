"""Playwright browser management."""

from playwright.async_api import async_playwright, Browser, BrowserContext, Page


class BrowserManager:
    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    async def start(self, headless: bool = True) -> Page:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        self.page = await self._context.new_page()
        return self.page

    async def navigate(self, url: str) -> None:
        await self.page.goto(url, wait_until="networkidle", timeout=30000)

    async def screenshot(self, path: str) -> None:
        await self.page.screenshot(path=path, full_page=False)

    async def get_title(self) -> str:
        return await self.page.title()

    async def get_url(self) -> str:
        return self.page.url

    async def get_accessibility_tree(self) -> dict | None:
        """Get accessibility tree via JS evaluation (compatible with Playwright 1.58+)."""
        try:
            return await self.page.evaluate('''() => {
                function walk(el) {
                    const tag = el.tagName?.toLowerCase() || '';
                    const role = el.getAttribute('role') || el.ariaRoleDescription || tag;
                    const name = el.getAttribute('aria-label')
                        || el.getAttribute('title')
                        || el.getAttribute('placeholder')
                        || ((['A','BUTTON','H1','H2','H3','H4','H5','H6','LABEL','INPUT','SELECT','TEXTAREA'].includes(el.tagName))
                            ? (el.textContent?.trim().slice(0,80) || '') : '');
                    const attrs = {};
                    if (el.href) attrs.href = el.href;
                    if (el.type) attrs.type = el.type;
                    if (el.value) attrs.value = el.value;
                    const children = [];
                    for (const child of el.children) {
                        const c = walk(child);
                        if (c) children.push(c);
                    }
                    // Map HTML tags to ARIA roles
                    let ariaRole = role;
                    const roleMap = {a:'link', button:'button', input:'textbox', select:'combobox',
                        textarea:'textbox', nav:'navigation', h1:'heading', h2:'heading',
                        h3:'heading', h4:'heading', h5:'heading', h6:'heading', ul:'list',
                        li:'listitem', img:'img', form:'form', main:'main', header:'banner',
                        footer:'contentinfo', aside:'complementary', dialog:'dialog',
                        details:'group', summary:'button', label:'label'};
                    if (roleMap[tag]) ariaRole = roleMap[tag];
                    if (el.type === 'checkbox') ariaRole = 'checkbox';
                    if (el.type === 'radio') ariaRole = 'radio';
                    if (el.type === 'password') ariaRole = 'textbox';
                    if (el.type === 'search') ariaRole = 'searchbox';

                    return {role: ariaRole, name: name, children: children.length ? children : undefined, ...attrs};
                }
                return walk(document.body);
            }''')
        except Exception:
            return None

    async def get_aria_snapshot(self) -> str:
        """Get aria snapshot string (Playwright 1.58+)."""
        try:
            return await self.page.locator('body').aria_snapshot()
        except Exception:
            return ""

    async def go_back(self) -> None:
        try:
            await self.page.go_back(wait_until="networkidle", timeout=10000)
        except Exception:
            pass  # Some pages can't go back; that's fine

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
