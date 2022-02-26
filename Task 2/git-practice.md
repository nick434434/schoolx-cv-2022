# Дополнительная практика git

## Ручная инициализация репозитория

Иногда модет сложиться ситуация, когда проект уже в процессе разработки, но для него не был создан удаленный репозиторий. В таких случаях вам понадобится знать, как это сделать. Попробуем на базовом примере.

* Создайте пустой удаленный репозиторий (без README и .gitignore) и скопируйте ссылку на него
* Создайте пустую папку с помощью `mkdir` и войдите в неё в терминале. Инициализируйте репозиторий, добавьте с помощью `git remote` новый ремоут и ссылку на него
* Создайте любой файл (или добавьте какой-нибудь ваш реализованный ранее проект для учёта его как в портфолио на github)
* Добавьте желаемые файлы в учет git
* Сделайте коммит вместе с сообщением о первом коммите: кратко опишите, что вы добавляете
* Запушьте изменения
* Зайдите на страницу удаленного репозитория и проверьте наличие вашего коммита и файлов
* Вы в танцах!

## Разрешение конфликтов

На примере любого проекта, который вам не жалко (можно создать новую пустышку для экспериментов) проведите следующую симуляцию реальной ситуации:

* Создайте две новых ветки на основе текущего состояния ветки `main`
* Сделайте некоторое множество изменений, добавьте ветку вместе с ними в удаленный репозиторий, сделайте мердж в `main`
* Переключившись на вторую созданную ветку, сделайте аналогичные действия, но так, чтобы были затронуты хотя бы некоторые места в коде, которые уже были изменены в первом случае, но сам новый код был другим (например, в одном случае - цикл, в другом - list comprehension и т.д.)
* Во время проведения мерджа второй ветки в `main` у вас должнвы появиться конфликты - разрешите их таким образом, чтобы
* * Фичи с обеих веток остались
* * В местах конфликта замены одного на другое осталась более "актуальная" версия
* * Сам код остался рабочим