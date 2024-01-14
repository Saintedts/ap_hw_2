import asyncio
import logging
from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import Command, StateFilter
from config_reader import config
from aiogram import F
from aiogram.fsm.context import FSMContext
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.fsm.state import StatesGroup, State
from model import MyModel

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value())
# Диспетчер
dp = Dispatcher()

router = Router()

model = MyModel(path='model/linear_regression_model.pkl')


class ScoreBotStates(StatesGroup):
    waiting_for_score = State()


class PredictBotStates(StatesGroup):
    waiting_for_height = State()


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    msg = f'''
Привет, {message.from_user.full_name}!
Рады видеть тебя в Tims ML BOT.
Вот что умеет бот:
    1. Предсказывать ваш рост на основе веса.
    2. Выводит оценки качества обученной модели.
    3. Вы можете оценить работу боту по 5-бальной шкале.
    4. Посмотреть среднюю оценку бота на основе других пользователей.
            '''
    await message.answer(msg)


# @dp.message(Command("predict"))
# async def cmd_predict(message: types.Message):
#     await message.answer("Hello!")


@router.message(Command("predict"), StateFilter(None))
async def cmd_predict(message: types.Message, state: FSMContext):

    await message.answer(
        "Введите ваш вес:",
        )
    await state.set_state(PredictBotStates.waiting_for_height)


@router.message(PredictBotStates.waiting_for_height,
                lambda message: message.text.isdigit() and
                1 <= int(message.text) <= 300)
async def handle_user_height(message: types.Message,
                             state: FSMContext):
    await state.update_data(chosen_height=message.text.lower())
    height = await state.get_data()
    prediction = model.predict([[int(height['chosen_height'])]])
    await message.answer(f"Ваш рост может быть равен {str(prediction)}")
    await state.clear()


@dp.message(Command("model_scores"))
async def cmd_model_scores(message: types.Message):
    await message.answer(str(model.model_quality()))


@dp.message(Command("bot_scores"))
async def cmd_bot_scores(message: types.Message, scores: list[int]):
    await message.answer(f"Средняя оценка бота: {sum(scores) / len(scores)}")


@router.message(Command("score_bot"), StateFilter(None))
async def cmd_score_bot(message: types.Message, state: FSMContext):

    builder = ReplyKeyboardBuilder()
    for i in range(1, 6):
        builder.add(types.KeyboardButton(text=str(i)))
    builder.adjust(5)
    await message.answer(
        "Выберите число:",
        reply_markup=builder.as_markup(resize_keyboard=True),
        )
    await state.set_state(ScoreBotStates.waiting_for_score)


@router.message(ScoreBotStates.waiting_for_score,
                lambda message: message.text.isdigit() and
                1 <= int(message.text) <= 5)
async def handle_user_score(message: types.Message,
                            scores: list[int],
                            state: FSMContext):
    await state.update_data(chosen_score=message.text.lower())
    curent_score = await state.get_data()
    scores.append(int(curent_score['chosen_score']))
    await message.answer(f"Вы выбрали число {curent_score['chosen_score']}")
    await state.clear()


@router.message(F.text, StateFilter(None))
async def echo_with_time(message: types.Message):
    await message.answer('Это не команда...')


# Запуск процесса поллинга новых апдейтов
async def main():
    dp.include_router(router)
    await dp.start_polling(bot, scores=[5])


if __name__ == "__main__":
    asyncio.run(main())
