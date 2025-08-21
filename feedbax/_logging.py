#! TODO: Uncomment when config is done
# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path

# from rich.logging import RichHandler

# from feedbax.config import LOGGING, PATHS


# def _remove_handlers(logger: logging.Logger, *, predicate) -> None:
#     """Remove and close all handlers on `logger` for which `predicate(handler)` is True."""
#     for h in list(logger.handlers):
#         if predicate(h):
#             logger.removeHandler(h)
#             h.close()


# def _make_rotating_handler(
#     path: Path, level: int, fmt: logging.Formatter
# ) -> RotatingFileHandler:
#     """Create a RotatingFileHandler writing to `path` at `level` with `fmt`."""
#     fh = RotatingFileHandler(
#         filename=str(path),
#         maxBytes=LOGGING.max_bytes,
#         backupCount=LOGGING.backup_count,
#         encoding="utf-8",
#     )
#     fh.setLevel(level)
#     fh.setFormatter(fmt)
#     return fh


# def _prune_foreign_file_handlers(central_dir: Path) -> None:
#     """
#     Strip any RotatingFileHandlers from *all* loggers whose baseFilename
#     does not live under central_dir.
#     """
#     for lg in logging.Logger.manager.loggerDict.values():
#         if not isinstance(lg, logging.Logger):
#             continue

#         for h in [h for h in lg.handlers if isinstance(h, RotatingFileHandler)]:
#             try:
#                 # if this handler writes *outside* of our central_dir, drop it
#                 if Path(h.baseFilename).resolve().parent != central_dir:
#                     lg.removeHandler(h)
#                     h.close()
#             except Exception:
#                 # ignore weird cases (e.g. missing baseFilename attr)
#                 pass


# _console_handler_pred = lambda h: (
#     isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
# )

# SESSION_START_BANNER = "―" * 20 + " NEW SESSION STARTED " + "―" * 20


# def enable_central_logging(
#     file_level: int | None = None,
#     console_level: int | None = None,
#     pkg_console_levels: dict[str, int] | None = None,
#     pkgs_own_files: dict[str, int] | None = None,
# ):
#     """
#     Configure `rich` console logs, and centralize all file logs into `PATHS.logs`.

#     Default arguments are taken from `LOGGING` config, if available.

#     Args:
#       file_level: minimum level for file output
#       console_level: minimum level for console output
#       pkg_console_levels: overrides console levels for specific packages
#       pkgs_own_files: write these packages to their own rotating files, at the specified level
#     """
#     # ─── 0) unpack defaults ──────────────────────────────────────────────────
#     file_lvl: int = file_level or LOGGING.file_level
#     console_lvl: int = console_level or LOGGING.console_level
#     pkg_console_lvls: dict[str, int] = pkg_console_levels or LOGGING.pkg_console_levels or {}
#     pkg_own_fs: dict[str, int] = pkgs_own_files or LOGGING.pkgs_own_files or {}

#     console_fmt = logging.Formatter(LOGGING.console_format_str)
#     file_fmt = logging.Formatter(LOGGING.file_format_str)

#     # ─── 1) prep central directory & prune stray handlers ───────────────────
#     logs_dir = Path(PATHS.logs).resolve()
#     logs_dir.mkdir(parents=True, exist_ok=True)
#     _prune_foreign_file_handlers(logs_dir)

#     # ─── 2) configure root logger ──────────────────────────────────────────
#     root = logging.getLogger()
#     root.setLevel(1)

#     root_log = logs_dir / "root.log"
#     root.addHandler(_make_rotating_handler(root_log, file_lvl, file_fmt))

#     # replace *any* old console streams with a single RichHandler
#     _remove_handlers(root, predicate=_console_handler_pred)
#     console_h = RichHandler(level=console_lvl)
#     console_h.setFormatter(console_fmt)
#     root.addHandler(console_h)

#     # ─── 3) per-package console overrides ───────────────────────────────────
#     for pkg, lvl in pkg_console_lvls.items():
#         lg = logging.getLogger(pkg)
#         _remove_handlers(lg, predicate=_console_handler_pred)

#         sh = RichHandler(level=lvl)
#         sh.setFormatter(console_fmt)
#         lg.addHandler(sh)
#         lg.propagate = True  # still bubble to root.log for files

#     # ─── 4) per-package own-file handlers ──────────────────────────────────
#     for pkg, lvl in pkg_own_fs.items():
#         lg = logging.getLogger(pkg)
#         _remove_handlers(lg, predicate=lambda h: isinstance(h, RotatingFileHandler))

#         pkg_log = logs_dir / f"{pkg}.log"
#         lg.addHandler(_make_rotating_handler(pkg_log, lvl, file_fmt))

#         lg.setLevel(min(lvl, file_lvl, console_lvl))
#         lg.propagate = False  # isolate to its own file

#     # ─── 5) let everything else bubble up to root ───────────────────────────
#     for name, lg in logging.Logger.manager.loggerDict.items():
#         if not isinstance(lg, logging.Logger) or name in pkg_own_fs:
#             continue
#         lg.setLevel(logging.NOTSET)
#         lg.propagate = True

#     # ─── 6) capture warnings into logging ──────────────────────────────────
#     logging.captureWarnings(True)

#     def _log_session_start_banner(lg):
#         if lg.level > logging.INFO:
#             lg.log(lg.level, SESSION_START_BANNER)
#         else:
#             lg.info(SESSION_START_BANNER)

#     # ─── final log of what we did ──────────────────────────────────────────
#     _log_session_start_banner(root)
#     root.info("Central logging enabled → %s", root_log)
#     for pkg, lvl in pkg_own_fs.items():
#         lg = logging.getLogger(pkg)
#         _log_session_start_banner(lg)
#         root.info("  • %s → %s (level %s)", pkg, logs_dir / f"{pkg}.log", lvl)