import pytest
from aiogram.filters import Command
from bot import echo_with_time, cmd_start, cmd_predict, cmd_score_bot, cmd_model_scores

from aiogram_tests import MockedBot
from aiogram_tests.handler import CallbackQueryHandler
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import CALLBACK_QUERY
from aiogram_tests.types.dataset import MESSAGE
from aiogram import Dispatcher


@pytest.mark.asyncio
async def test_message_handler():
    requester = MockedBot(MessageHandler(echo_with_time))
    calls = await requester.query(MESSAGE.as_object(text="some random text: skjndglskdf;skdf;smdfl"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == 'Это не команда...'


@pytest.mark.asyncio
async def test_cmd_start():
    requester = MockedBot(MessageHandler(cmd_start))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    print(answer_message)
    assert answer_message == '''
Привет, FirstName LastName!
Рады видеть тебя в Tims ML BOT.
Вот что умеет бот:
    1. Предсказывать ваш рост на основе веса.
    2. Выводит оценки качества обученной модели.
    3. Вы можете оценить работу боту по 5-бальной шкале.
    4. Посмотреть среднюю оценку бота на основе других пользователей.
            '''


@pytest.mark.asyncio
async def test_cmd_predict():
    requester = MockedBot(MessageHandler(cmd_predict))
    calls = await requester.query(MESSAGE.as_object(text="some random text, not digits: skjndglskdf;skdf;smdfl"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == 'Введите ваш вес:'
    # ТК state не изменился, то не переходим к обработчику прогнозирования ==> сообщение будет прежним


@pytest.mark.asyncio
async def test_cmd_model_scores():
    requester = MockedBot(MessageHandler(cmd_model_scores))
    calls = await requester.query(MESSAGE.as_object(text="/model_scores"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == str({
            'R2': 0.8578,
            'MSE': 13.6281,
            'MAX_ERROR': 14.2192
        })
