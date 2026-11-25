"""Test the interactive incident queue in a local Chromium browser."""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium")
    args=parser.parse_args()
    results=Path("results")
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,executable_path=args.chromium)
        page=browser.new_page(viewport={"width":1130,"height":790})
        errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto((results/"alert-triage.html").resolve().as_uri())
        assert page.locator("#cases tr").count()==24
        assert page.locator("#attributions .attr").count()==3
        page.locator("#sort").select_option("index")
        page.get_by_role("button",name="Case 03",exact=True).click()
        assert page.locator("#title").inner_text()=="Case 03"
        page.locator("#model").select_option("mlp")
        page.locator("#method").select_option("mlp_lime")
        assert "Surrogate probability" in page.locator("#units").inner_text()
        page.locator("#method").select_option("mlp_ig")
        assert "logit contributions" in page.locator("#units").inner_text()
        page.locator("#sort").select_option("risk")
        page.locator("#cases button").first.click()
        page.screenshot(path=results/"alert-triage.png")
        page.set_viewport_size({"width":390,"height":844})
        assert page.locator("#queue").bounding_box()["width"]<390
        assert not errors,errors
        browser.close()
    print("Triage queue, explanation controls and mobile layout passed.")


if __name__=="__main__":
    main()
