"""019 ETF 日线数据同步 — 主程序入口。

独立于 004_sequoia-x 项目，通过独立 cron 调度（交易日 20:00）。

运行模式：
  python main.py                    # 标准模式：ETF 列表 + 日线 + 指数（20:00 后）
  python main.py --sync-only        # 仅同步数据（跳过 ETF 列表更新）
  python main.py --force            # 跳过交易日/时间门控检查
  python main.py --backfill         # 全量回填：从 start_date 起拉取所有 ETF
  python main.py --list-only        # 仅更新 ETF 列表
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from etf_sync.config import get_settings
from etf_sync.logger import get_logger
from etf_sync.notify import (
    push_backfill_summary,
    push_data_sync_summary,
    push_error_alert,
    push_skip_notice,
    push_sync_summary,
)
from etf_sync.sync import ETFSync


def main() -> None:
    parser = argparse.ArgumentParser(description="019 ETF 日线数据同步")
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="仅执行数据同步（跳过 ETF 列表更新）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制模式：跳过交易日/时间门控检查",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="回填模式：全量拉取所有 ETF 历史数据（从 start_date 起）",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="仅同步 ETF 列表（不拉取日线数据）",
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
        logger = get_logger(__name__)
        sync_mgr = ETFSync(settings)
        t0: float = time.time()

        if args.backfill:
            logger.info("=== 回填模式 ===")
            logger.info("先同步 ETF 列表...")
            r1 = sync_mgr.sync_etf_list()
            logger.info("开始全量回填 ETF 日线（force=True）...")
            r2 = sync_mgr.sync_etf_daily(force=True)
            elapsed = time.time() - t0
            logger.info(
                f"ETF 回填: status={r2['status']}, "
                f"{r2.get('etf_count', 0)} 只, {elapsed:.0f}s"
            )
            logger.info("开始回填指数日线（force=True）...")
            r3 = sync_mgr.sync_index_daily(force=True)
            total_elapsed = time.time() - t0
            logger.info(f"指数回填: {r3.get('index_count', 0)} 个")
            logger.info(f"回填总耗时: {total_elapsed:.0f}s")
            # 推送回填完成通知
            push_backfill_summary(settings, r1, r2, r3, total_elapsed)
            return

        if args.list_only:
            logger.info("=== 仅同步 ETF 列表 ===")
            result = sync_mgr.sync_etf_list()
            elapsed = time.time() - t0
            logger.info(
                f"ETF 列表: 共 {result.get('total', 0)} 只, "
                f"新增 {len(result.get('new_listed', []))} 只, "
                f"退市 {len(result.get('delisted', []))} 只, "
                f"耗时 {elapsed:.0f}s"
            )
            return

        if args.sync_only:
            logger.info("=== 仅同步日线数据 ===")
            r2 = sync_mgr.sync_etf_daily(force=args.force)
            r3 = sync_mgr.sync_index_daily(force=args.force)
            elapsed = time.time() - t0
            logger.info(
                f"同步完成: ETF {r2.get('etf_count', 0)} 只, "
                f"指数 {r3.get('index_count', 0)} 个, "
                f"耗时 {elapsed:.0f}s"
            )
            # 推送同步完成通知
            push_data_sync_summary(settings, r2, r3, elapsed)
            return

        # ═══════════════════════════════════════════════
        #  标准模式（run_full 管线）
        # ═══════════════════════════════════════════════
        logger.info("=== 标准模式 ===")
        result = sync_mgr.run_full()
        elapsed = time.time() - t0
        logger.info(
            f"管线完成: status={result['status']}, 耗时 {elapsed:.0f}s"
        )

    except Exception:
        try:
            _logger = get_logger(__name__)
            _logger.exception("主流程未捕获异常，程序终止")
        except Exception:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
