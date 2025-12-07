# Rail MAS (SPADE) - Decentralized (Contract Net) Freight Scheduling

**Без центрального координатора.** Каждая станция-инициатор (origin заказа) сама объявляет конкурс (CFP) по контрактной сети (CNP), собирает предложения, выбирает победителя и публикует результат. Любая станция может участвовать, учитывая стоимость перегонки своих локомотивов/вагонов в точку отправления и затем перевозки под нагрузкой.

## Роли и сообщения
- **StationAgent**: единственный тип агента. Умеет:
  - инициировать CFP по всем заказам, у которых `from == station_id`;
  - отвечать PROPOSE на CFP других станций, если может обеспечить локомотив/вагоны (с учётом перегонки);
  - принимать ACCEPT/REJECT; победитель фиксирует план и публикует INFORM (broadcast).
- Сообщения: `CFP`, `PROPOSE`, `ACCEPT`, `REJECT`, `INFORM` (JSON payload).

## Как считается предложение
- Проверяем достаточность вагонов нужного типа (в т, суммарная вместимость >= tonnage).
- Проверяем локомотив по **max_tonnage >= суммарной вместимости**.
- Стоимость = `dist_empty * cost_per_km (перегон пустых)` + `dist_loaded * cost_per_km`.
- Время = `time_empty + time_loaded`. Отправление принимаем за `t=0` для демо.

## Запуск (пример с 5 станциями)
1) Установить зависимости:
```bash
pip install -r requirements.txt
```
2) Поднять XMPP и завести пользователей: `station-alpha`, `station-bravo`, `station-charlie`, `station-delta`, `station-echo`.
3) На любой машине запустить агентов станций (можно на разных узлах):
```bash
python run_station_p2p.py --jid station-alpha@26.9.58.34 --password alpha --station_id ALPHA --data data/input.json
python run_station_p2p.py --jid station-bravo@26.9.58.34 --password bravo --station_id BRAVO --data data/input.json
python run_station_p2p.py --jid station-charlie@26.9.58.34 --password charlie --station_id CHARLIE --data data/input.json
python run_station_p2p.py --jid station-delta@26.9.58.34 --password delta --station_id DELTA --data data/input.json
python run_station_p2p.py --jid station-echo@26.9.58.34 --password echo --station_id ECHO --data data/input.json

python run_station_p2p.py --jid ... --password ... --station_id ECHO --data data/input.json --is_last

```
4) По окончании торга инициаторы пишут итоговые планы в `output/schedule.csv|json` (каждый агент ведёт **свой** файл; при желании можно объединить потом).

## Файл данных
`data/input.json` — станционные парки, граф, заказы и JID всех станций.

## Улучшения
- Добавить дедлайны и штрафы в функцию стоимости.
- Ввести журнал занятости локомотивов/вагонов и окна времени.
- Использовать MUC или PubSub для более удобной широковещательной рассылки.
