from io import StringIO

def dfInfoLogger(logger, df) -> None:
    buf = StringIO()
    df.info(buf=buf)
    logger.info(f">> info()\r\n{buf.getvalue()}")
    