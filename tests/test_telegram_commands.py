"""Telegram komut ayrıştırma + dispatch testleri (saf, ağ yok)."""
from nihai.notifier.telegram_bot import CommandListener, parse_command


def test_parse_basic():
    assert parse_command("/add nvda") == ("add", "NVDA")
    assert parse_command("/status") == ("status", None)
    assert parse_command("/remove TSLA") == ("remove", "TSLA")


def test_parse_botname_suffix_stripped():
    assert parse_command("/add@MyTradeBot nvda") == ("add", "NVDA")


def test_parse_case_and_whitespace():
    assert parse_command("  /ADD   nvda  ") == ("add", "NVDA")


def test_parse_non_command():
    assert parse_command("merhaba") == (None, None)
    assert parse_command("") == (None, None)
    assert parse_command("/") == (None, None)


def test_dispatch_calls_handler():
    seen = {}
    def add(arg):
        seen["arg"] = arg
        return f"ok {arg}"
    cl = CommandListener({"add": add}, chat_id="123")
    assert cl.dispatch("/add nvda") == "ok NVDA"
    assert seen["arg"] == "NVDA"


def test_dispatch_unknown_command():
    cl = CommandListener({}, chat_id="123")
    out = cl.dispatch("/foo")
    assert "Bilinmeyen komut" in out and "/foo" in out


def test_dispatch_non_command_returns_none():
    cl = CommandListener({"add": lambda a: "x"}, chat_id="123")
    assert cl.dispatch("selam") is None


def test_dispatch_handler_exception_caught():
    def boom(arg):
        raise ValueError("patladı")
    cl = CommandListener({"add": boom}, chat_id="123")
    assert cl.dispatch("/add x").startswith("Hata:")
