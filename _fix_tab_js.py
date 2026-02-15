"""Fix tab switch JS: replace iframe-scoped listeners with parent-document listeners."""
import re

filepath = r"c:\Users\ufuka\Desktop\olasilikquiz1\streamlit_app.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace the entire _TAB_SWITCH_JS block
old_pattern = r'_TAB_SWITCH_JS = """.*?"""'
new_js = r'''_TAB_SWITCH_JS = """
<script>
(function() {
    // Parent document = gercek Streamlit sayfasi (iframe degil)
    var parentDoc = window.parent.document;
    var parentWin = window.parent;

    // Dinleyicinin birden fazla kez eklenmesini engelle
    if (parentWin.__quizTabGuardActive) return;
    parentWin.__quizTabGuardActive = true;

    // Parent document uzerinde visibilitychange dinle
    parentDoc.addEventListener('visibilitychange', function() {
        if (parentDoc.visibilityState === 'hidden') {
            // Ogrenci baska sekmeye/uygulamaya gecti!
            var now = new Date().toISOString();
            try {
                parentWin.localStorage.setItem('QUIZ_TAB_VIOLATION', JSON.stringify({
                    violated: true,
                    timestamp: now
                }));
            } catch(e) {}

            // Sayfayi tab_violation query parametresiyle yeniden yukle
            var url = new URL(parentWin.location.href);
            url.searchParams.set('tab_violation', '1');
            url.searchParams.set('vt', Date.now().toString());
            parentWin.location.replace(url.toString());
        }
    });
})();
</script>
"""'''

match = re.search(old_pattern, content, re.DOTALL)
if match:
    content = content[:match.start()] + new_js + content[match.end():]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("BASARILI: Tab switch JS guncellendi!")
    print(f"Eski uzunluk: {len(match.group())} karakter")
    print(f"Yeni uzunluk: {len(new_js)} karakter")
else:
    print("HATA: _TAB_SWITCH_JS blogu bulunamadi!")
