"""Потоковый диалоговый ассистент приёмной комиссии — backend (`001-streaming-dialogue`).

Один python-процесс, asyncio (NFR-06). Точка входа — `backend.app:app`; вся
конфигурация читается один раз в `backend.config` (FR-32).
"""
