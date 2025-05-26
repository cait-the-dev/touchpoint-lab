from src.bot_filter import is_bot

def test_bot_rules():
    assert not is_bot(3, 2, "alice@gmail.com")
    assert not is_bot(55, 10, "bob@yahoo.com")
    assert is_bot(1, 0, "foo@tempmail.com")
    assert is_bot(0, 0, "bar@mailinator.com")
    assert is_bot(300, 290, "charlie@proton.me")
    assert is_bot(80, 80, "spam@10minutemail.com")
    assert not is_bot(0, 0, "")
    assert not is_bot(None, None, None)

    sample = [
        (3, 2, "alice@gmail.com"),
        (55, 10, "bob@yahoo.com"),
        (1, 0, "foo@tempmail.com"),
        (0, 0, "bar@mailinator.com"),
        (300, 290, "charlie@proton.me"),
        (80, 80, "spam@10minutemail.com"),
        (0, 0, "dave@gmail.com"),
        (12, 5, "eve@hotmail.com"),
        (10, 60, "frank@outlook.com"),
        (0, 0, ""),
    ]
    bot_flags = [is_bot(o, c, e) for o, c, e in sample]
    assert sum(bot_flags) == 4
