import requests

def trace_vaa3d(URL, tif_file, swc_file):
    assert isinstance(tif_file, str) and isinstance(swc_file, str)
    try:
        with open(tif_file, "rb") as f:
            files = {"file": f}
            r = requests.post(URL, files=files, timeout=1200)

        if r.status_code != 200:
            print(f"[FAIL] {tif_file} -> status {r.status_code}")
            return False
        else:
            with open(swc_file, "wb") as out:
                out.write(r.content)
            return True
    except Exception as e:
        print(f"[ERROR] {tif_file}: {e}")
        return False