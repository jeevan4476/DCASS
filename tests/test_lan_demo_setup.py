from scripts.demo.lan_demo_setup import get_local_ip, check_port_open

def test_get_local_ip_returns_valid_string():
    ip = get_local_ip()
    assert isinstance(ip, str)
    parts = ip.split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert 0 <= int(part) <= 255

def test_check_port_open_nonexistent():
    # An unallocated high port should return False
    assert check_port_open("127.0.0.1", 59999) is False
