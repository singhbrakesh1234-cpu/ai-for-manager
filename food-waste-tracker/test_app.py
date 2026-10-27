# Automatic check of app.py without a browser:  python test_app.py
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception
print("Tabs:", [t.label for t in at.tabs])
btn = [b for b in at.button if "Check waste risk" in str(b.label)][0]
btn.click().run(); assert not at.exception, at.exception
print("Add-item check:", [w.value[:70] for w in at.warning] or [s.value[:70] for s in at.success])
hh = at.selectbox[0]
for opt in hh.options[:4]:
    hh.select(opt).run(); assert not at.exception, at.exception; hh = at.selectbox[0]
print("ALL CHECKS PASSED")
