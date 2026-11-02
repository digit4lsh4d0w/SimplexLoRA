# Техническое задание: наблюдаемость и экспорт калибровки SimplexLoRA

Статус: проект ТЗ

Целевая версия схемы артефактов: `1.0`

Область: ядро `SimplexAdamW`, PEFT-плагин, CLI экспериментов и инструменты анализа

## 1. Цель

Сделать фазу калибровки SimplexLoRA воспроизводимой и наблюдаемой. После
калибровочного прогона пользователь должен получить:

1. понятный человеку отчёт о том, как менялось распределение важности;
2. машинно-читаемый и версионированный план слоёв;
3. достаточно диагностических данных, чтобы отличить устойчивое распределение
   от результата случайной инициализации или неудачной калибровочной выборки;
4. готовый файл параметров для передачи распределения рангов в обычный LoRA или
   Unsloth без импорта и использования custom-слоёв SimplexLoRA в downstream-
   пайплайне;
5. гарантию, что информация сохраняется до того, как
   `_final_lora_rank_update()` переинициализирует `lora_A`/`lora_B` и зафиксирует
   `lora_weight = 1`.

Результат калибровки не обязан быть обученным адаптером. Основной handoff-артефакт
— это распределение рангов и связанные с ним метрики. Качество downstream-модели
оценивается отдельно после обычного LoRA-обучения.

## 2. Текущее состояние и проблемы

Сейчас `SimplexAdamW` уже вычисляет веса симплекса, ранги и список выбранных
модулей, но:

- выводит в stdout только ранги и числовые индексы параметров;
- не сохраняет стабильные имена модулей;
- не сохраняет исходный вес и результат каждой проекции;
- не пишет историю loss и градиентов во время калибровки;
- не экспортирует `layer_plan.json`;
- не сохраняет промежуточные данные до финального сброса `lora_A`/`lora_B`;
- не проверяет, что записанный ранг действительно применён к модели;
- не позволяет сравнить калибровки с разными seed и подвыборками;
- не формирует конфигурацию `target_modules`, `rank_pattern` и `alpha_pattern`
  для Unsloth.

`training_summary.json` содержит только итоговые метрики Trainer и не является
достаточным handoff-артефактом.

## 3. Границы работ

### 3.1. Входит в ТЗ

- сбор событий калибровки непосредственно из `SimplexAdamW`;
- стабильная идентификация модулей и адаптера;
- метрики концентрации, покрытия и изменения распределения;
- промежуточные и финальные структурированные артефакты;
- человеко-читаемый Markdown-отчёт и краткая сводка в stdout;
- отдельный режим «только калибровка»;
- экспорт `layer_plan.json` до финального обнуления важности;
- проверка соответствия плана реальной модели;
- отдельный экспорт конфигурации для Unsloth/PEFT;
- сравнение нескольких запусков с разными seed;
- unit-тесты на CPU и интеграционные тесты на уменьшенной модели;
- обработка ошибок, атомарная запись и правила частично завершённых запусков.

### 3.2. Не входит в ТЗ

- реализация обучения внутри Unsloth;
- добавление Unsloth как обязательной зависимости;
- перенос `lora_A`/`lora_B` из SimplexLoRA в Unsloth;
- автоматический выбор лучших гиперпараметров калибровки;
- доказательство причинной связи выбранных слоёв с качеством модели;
- генерация или исполнение декомпилированного кода;
- поддержка нескольких одновременно калибруемых адаптеров;
- распределённая калибровка на нескольких GPU.

Если несколько адаптеров или распределённый запуск понадобятся позже, для них
должен быть отдельный этап проектирования. До этого такие конфигурации должны
завершаться явной ошибкой, а не создавать неполные или неоднозначные артефакты.

## 4. Термины и инварианты

**Калибровочный прогон** — часть обучения до окончательной проекции симплекса,
в которой оптимизируются `lora_weight`.

**Проекция** — событие каждые `fat_step` optimizer steps. На нём веса
проецируются на симплекс, вычисляются ранги и множество активных слоёв.

**Финальная проекция** — проекция, после которой `max_fat_steps` достигает нуля,
а `_final_lora_rank_update()` фиксирует окончательные ранги.

**План слоёв** — неизменяемый снимок финальной проекции: имя модуля, вес
симплекса, ранг, alpha/scaling, признаки выбора и метаданные модели.

**Optimizer step** — один вызов `optimizer.step()` после всех micro-batch,
входящих в текущий шаг gradient accumulation.

Обязательные инварианты:

- порядок слоёв определяется стабильным именем модуля, а не порядком словаря
  параметров;
- вес важности и ранг берутся из той же проекции, которая применяется к модели;
- `rank_i = ceil(weight_i * default_lora_rank)`;
- финальный план сохраняется до вызова `_final_lora_rank_update()`;
- веса и ранги никогда не восстанавливаются обратным вычислением из
  `lora_A`/`lora_B`;
- события калибровки не изменяют параметры, RNG, gradients или scheduler;
- JSON не содержит `NaN`, `Infinity`, tensor, numpy scalar или объект с
  нестабильной сериализацией;
- все пути внутри артефактов относительны каталогу запуска;
- имена модулей экранируются при превращении в regex.

## 5. Целевой поток выполнения

1. Модель загружается и к ней применяется SimplexLoRA.
2. Создаётся реестр всех `SimplexLoraLinear` с именами и размерностями.
3. Создаётся `CalibrationRecorder`.
4. Trainer записывает loss каждого optimizer step.
5. `SimplexAdamW` на каждом шаге передаёт observer-у loss и нормы градиентов.
6. На каждой проекции observer получает точные веса до и после шага, результат
   проекции, ранги и маску выбранных слоёв.
7. Промежуточные события сразу попадают в append-only JSONL.
8. После финальной проекции, но до переинициализации A/B:

   - создаётся `LayerPlan`;
   - валидируется схема;
   - сохраняются `layer_plan.json`, CSV, Markdown и handoff-файл;
   - выполняется сверка плана с моделью.

9. При `--calibration-only` обучение останавливается.
10. При обычном обучении финальный сброс и последующее LoRA-обучение продолжаются.
11. В конце обновляется manifest со статусом и контрольными суммами артефактов.

## 6. Артефакты

### 6.1. Расположение

По умолчанию артефакты создаются в:

```text
<output-dir>/calibration/
├── manifest.json
├── steps.jsonl
├── projections.jsonl
├── calibration_layers.csv
├── calibration_report.md
├── layer_plan.json
└── unsloth_handoff.json
```

JSONL-файлы создаются инкрементально. Остальные файлы записываются через
временный файл в том же каталоге и атомарный `replace`.

### 6.2. `manifest.json`

Главный структурированный документ запуска. Обязательные поля:

- `schema_version`;
- `run_id`;
- `status`: `running`, `completed`, `partial` или `failed`;
- `created_at`, `updated_at` в UTC ISO 8601;
- версия `simplexlora`, commit hash, версии Python, PyTorch, PEFT, Transformers,
  bitsandbytes и ускорителя;
- `model_id`, `model_revision`, класс модели, устройство;
- имя адаптера и PEFT-конфигурация;
- `default_lora_rank`, `lora_alpha`, `fat_step`, `max_fat_steps`, `weight_lr`,
  `weight_decay`;
- число optimizer steps, effective batch size, gradient accumulation;
- описание датасета или свободный object `data`;
- список выполненных проекций и путей к связанным артефактам;
- `calibration_completed`;
- `stop_reason`;
- warnings;
- пути и SHA-256 всех завершённых артефактов.

`run_id` должен быть детерминированным от seed, model id/revision, adapter name и
ключевых параметров оптимизатора либо явно передан пользователем.

### 6.3. `steps.jsonl`

Одна строка на optimizer step в пределах калибровки. Минимальные поля:

- `schema_version`;
- `step`;
- `epoch`;
- `loss`;
- `learning_rate`;
- `lora_grad_norm`;
- `weight_grad_norm`;
- `elapsed_seconds`;
- `is_projection_step`.

`loss` — значение, которое Trainer использует для своего training loss на данном
optimizer step, с учётом gradient accumulation. Если точное значение получить
нельзя, запись должна содержать `loss: null` и warning
`loss_unavailable`, а не подменять его последним micro-batch loss.

### 6.4. `projections.jsonl`

Одна строка на проекцию. Поля:

- `schema_version`;
- `projection_index`;
- `step`;
- `is_final_projection`;
- `loss_at_projection`;
- `mean_loss_since_previous_projection`;
- `weights_before_gradient_step`;
- `weights_after_gradient_step`;
- `projected_weights`;
- `ranks`;
- `selected_mask`;
- агрегированные метрики из `CalibrationMetrics`;
- список per-layer записей либо ссылка на срез `calibration_layers.csv`.

Для `1.0` per-layer записи необходимо хранить прямо в JSONL, чтобы файл оставался
самодостаточным. Начальный размер: около
`число_проекций × число_модулей` записей.

### 6.5. `calibration_layers.csv`

Денормализованная таблица для анализа в pandas, LibreOffice или Excel. Одна
строка соответствует паре «проекция, модуль»:

```text
projection_index,step,module_name,layer_index,component,
weight_before,weight_after,weight_projected,normalized_importance,
rank,previous_rank,rank_delta,selected,lora_alpha,scaling
```

CSV должен быть UTF-8, с `\n` и стабильным порядком сортировки:
`projection_index`, затем `layer_index`, затем `component`, затем `module_name`.

### 6.6. `calibration_report.md`

Человеко-читаемый отчёт. Обязательные разделы:

1. Краткое резюме: число шагов, проекций, финальный ранг-бюджет, выбранные слои.
2. Параметры запуска и fingerprint модели/данных.
3. Таблица метрик по каждой проекции.
4. Top-K модулей финальной проекции с весом, рангом, alpha и scaling.
5. Изменение рангов между проекциями для top-K.
6. Активность по компонентам: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `gate_proj`, `up_proj`, `down_proj`.
7. Слои с нулевым рангом.
8. Диагностика концентрации и предупреждения.
9. Сводка для Unsloth handoff.
10. Пути ко всем артефактам и ограничения интерпретации.

Отчёт не должен содержать полную матрицу на сотни строк. Полные данные остаются
в JSONL и CSV.

### 6.7. `layer_plan.json`

Стабильный handoff-формат. Минимальная структура:

```json
{
  "schema_version": "1.0",
  "run_id": "string",
  "model": {
    "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "revision": "string",
    "architecture": "Qwen2ForCausalLM"
  },
  "adapter": {
    "name": "default",
    "projection_step": 50,
    "default_lora_rank": 64,
    "lora_alpha": 128,
    "total_rank": 0,
    "selected_module_count": 0
  },
  "modules": [
    {
      "module_name": "model.layers.0.self_attn.q_proj",
      "peft_module_name": "base_model.model.model.layers.0.self_attn.q_proj",
      "layer_index": 0,
      "component": "q_proj",
      "in_features": 3584,
      "out_features": 3584,
      "simplex_weight": 0.0,
      "normalized_importance": 0.0,
      "rank": 0,
      "selected": false,
      "lora_alpha": 128,
      "scaling": 0.0
    }
  ]
}
```

План содержит все участвовавшие модули, включая `rank = 0`. Это важно для аудита
и сравнения запусков. Downstream-код обязан явно отфильтровывать нулевые ранги.

### 6.8. `unsloth_handoff.json`

Отдельный производный документ, не требующий установленного Unsloth:

- `source_plan_run_id`;
- `selection_policy`;
- `alpha_mode`;
- `rank_floor`;
- `target_modules`;
- `rank_pattern`;
- `alpha_pattern`;
- `excluded_modules`;
- `requested_rank_total`, `applied_rank_total`;
- `estimated_trainable_parameters`;
- warnings.

Полные имена модулей в regex должны быть экранированы и начинаться с `^`.
PEFT добавляет ограничение конца совпадения автоматически; генератор всё равно должен
формировать однозначные шаблоны.

## 7. Основные структуры данных

### 7.1. `LayerIdentity`

Смысл: однозначная запись о калибруемом LoRA-модуле.

Поля:

- `module_name` — каноническое имя относительно базовой модели;
- `peft_module_name` — полное имя внутри текущей PEFT-модели;
- `module_index` — стабильный целочисленный индекс в registry;
- `layer_index` — номер transformer layer;
- `component` — имя целевой линейной проекции;
- `in_features`, `out_features`;
- `initial_rank`;
- `lora_alpha`;
- `scaling`.

### 7.2. `ProjectionEvent`

Смысл: неизменяемый снимок одной проекции, переданный из оптимизатора.

Поля:

- `step`, `projection_index`;
- `is_final_projection`;
- `weights_before_step`;
- `weights_after_gradient_step`;
- `weights_projected`;
- `ranks`;
- `selected_mask`;
- `loss_at_projection`;
- `loss_window`;
- `weight_grad_norms`;
- `lora_grad_norms`;
- `created_at`.

Все tensor-значения должны быть предварительно detached, перенесены в CPU и
преобразованы в стабильные Python-типы.

### 7.3. `CalibrationMetrics`

Смысл: агрегированное описание распределения важности.

Обязательные поля:

- `module_count`;
- `selected_count`;
- `zero_rank_count`;
- `total_rank`;
- `rank_utilization`;
- `entropy`;
- `normalized_entropy`;
- `effective_module_count`;
- `hhi`;
- `inverse_hhi`;
- `top_1_share`;
- `top_5_share`;
- `top_k_share`;
- `component_shares`;
- `rank_by_component`;
- `max_rank`;
- `mean_selected_rank`.

### 7.4. `LayerPlan`

Смысл: замороженный и проверенный результат финальной проекции для передачи в
другой тренер.

Содержит metadata, adapter metadata и отсортированный список
`LayerPlanEntry`.

## 8. Крупные функции

### 8.1. `build_layer_registry`

Сигнатура:

```python
build_layer_registry(
    model,
    adapter_name: str,
    *,
    target_modules: Sequence[str] | None = None,
) -> LayerRegistry
```

Смысл:

- один раз обнаружить все `SimplexLoraLinear`;
- связать параметры оптимизатора с человекопонятными именами;
- устранить зависимость артефактов от нестабильного порядка
  `optimizer.param_groups`;
- заранее проверить геометрию и уникальность модулей.

Вход:

- PEFT-модель с зарегистрированным SimplexLoRA;
- имя активного адаптера;
- необязательный фильтр target modules.

Выход:

- immutable `LayerRegistry`;
- tuple `LayerIdentity`;
- отображения `module_name -> module_index` и
  `module_index -> LayerIdentity`;
- явный признак, что registry построен для одного адаптера.

Гарантии и ошибки:

- registry не меняет модель;
- порядок детерминирован: `layer_index`, `component`, `module_name`;
- отсутствие адаптера, дубли имён, нулевое число слоёв и несовпадение
  `lora_A`/`lora_B` dimensions являются ошибкой;
- наличие более одного калибруемого адаптера в версии `1.0` является ошибкой;
- каноническое имя обязано разрешаться обратно в модуль базовой модели.

### 8.2. `CalibrationRecorder`

Сигнатура:

```python
CalibrationRecorder(
    *,
    registry: LayerRegistry,
    output_dir: Path,
    run_metadata: CalibrationRunMetadata,
    top_k: int,
    fail_on_write_error: bool = True,
)
```

Смысл:

- быть единственным владельцем состояния калибровочного прогона;
- принимать события от оптимизатора и Trainer;
- проверять их до записи;
- инкрементально сохранять JSONL;
- строить финальные документы и сводки.

Вход:

- registry;
- каталог артефактов;
- версионированные метаданные запуска;
- размер Top-K;
- политика ошибок записи.

Выход:

- готовый к использованию recorder;
- публичные методы `record_step`, `record_projection`, `finalize`,
  `write_artifacts`, `close`.

Гарантии и ошибки:

- recorder не владеет tensor-объектами и не держит ссылки на параметры;
- `record_step` вызывается ровно один раз на optimizer step;
- `record_projection` вызывается ровно один раз на проекцию;
- повторная запись того же step или projection index является ошибкой;
- при `fail_on_write_error=True` ошибка артефакта останавливает обучение;
- даже при ошибке recorder помечает manifest как `failed` или `partial`.

### 8.3. `CalibrationRecorder.record_step`

Сигнатура:

```python
record_step(
    *,
    step: int,
    epoch: float | None,
    loss: float | None,
    learning_rate: float,
    lora_grad_norm: float,
    weight_grad_norm: float,
    elapsed_seconds: float,
    is_projection_step: bool,
) -> StepRecord
```

Смысл:

- сохранить динамику loss и градиентов внутри калибровочной фазы;
- дать возможность увидеть расходимость, взрыв градиентов или вырождение
  оптимизации до финальной проекции.

Вход:

- номер optimizer step;
- необязательный epoch;
- training loss, уже агрегированный по всем micro-batch этого шага;
- learning rate;
- нормы LoRA- и weight-градиентов после clipping;
- длительность шага;
- признак проекции.

Выход:

- нормализованный `StepRecord`, добавленный в память и `steps.jsonl`.

Гарантии и ошибки:

- числа должны быть конечными;
- при недоступном loss допускается `None` только вместе с warning;
- запись flush-ится до начала следующего optimizer step;
- ошибка сериализации не должна маскироваться молчанием.

### 8.4. `CalibrationRecorder.record_projection`

Сигнатура:

```python
record_projection(event: ProjectionEvent) -> ProjectionSnapshot
```

Смысл:

- зафиксировать точные вычисления, выполненные `SimplexAdamW`;
- посчитать метрики концентрации;
- сохранить состояние до финального сброса адаптеров.

Вход:

- `ProjectionEvent` с массивами длины `len(registry)`.

Выход:

- `ProjectionSnapshot` с per-layer записями и `CalibrationMetrics`;
- новая строка в `projections.jsonl`;
- строки текущей проекции в CSV-представлении.

Гарантии и ошибки:

- длины всех массивов совпадают с registry;
- `weights_projected >= 0`, сумма положительна;
- ранги равны `ceil(weights_projected * default_lora_rank)`;
- selected mask равна `rank > 0`;
- rank и weights не пересчитываются из другой формулы;
- `is_final_projection` устанавливается оптимизатором, а не угадывается по
  `max_steps`;
- наличие финальной проекции фиксируется до `_final_lora_rank_update()`.

### 8.5. Интеграция с Trainer: `SimplexCalibrationTrainer.training_step`

Сигнатура должна сохранять контракт `transformers.Trainer.training_step`:

```python
training_step(model, inputs, num_items_in_batch=None) -> torch.Tensor
```

Смысл:

- получить ровно тот loss, который Trainer использует для текущего шага;
- накопить его по micro-batch внутри gradient accumulation;
- передать итог оптимизатору до вызова `optimizer.step()`.

Вход:

- стандартный вход `Trainer.training_step`.

Выход:

- неизменённый `loss.detach()`, возвращаемый базовым Trainer;
- побочный эффект: `optimizer.add_microbatch_loss(loss.detach())`.

Гарантии и ошибки:

- оптимизация и backward не меняются;
- accumulated loss сбрасывается после каждого `optimizer.step()`;
- нельзя подменять loss средним только одного последнего micro-batch;
- при нестандартном Trainer без такой интеграции loss в отчёте должен быть
  `null`, а `loss_unavailable` — явно указан.

### 8.6. `compute_calibration_metrics`

Сигнатура:

```python
compute_calibration_metrics(
    weights: Sequence[float],
    ranks: Sequence[int],
    *,
    default_lora_rank: int,
    components: Sequence[str],
    top_k: int,
) -> CalibrationMetrics
```

Смысл:

- превратить распределение весов в набор сравнимых числовых характеристик;
- отделить важность слоя от нормы его градиента;
- сделать отчёты разных запусков сопоставимыми.

Вход:

- projected weights симплекса;
- итоговые ranks;
- исходный rank budget;
- типы компонентов;
- значение K для top-k share.

Выход:

- `CalibrationMetrics`.

Формулы:

- `p_i = weight_i / sum(weights)`;
- `entropy = -sum(p_i * log(p_i))` только для `p_i > 0`;
- `normalized_entropy = entropy / log(module_count)`;
- `effective_module_count = exp(entropy)`;
- `hhi = sum(p_i ** 2)`;
- `inverse_hhi = 1 / hhi`;
- `top_k_share = sum(top_k значений p_i)`;
- `rank_utilization = total_rank / (default_lora_rank * module_count)`.

Гарантии и ошибки:

- пустые входы и несовпадение длин запрещены;
- отрицательные weight, NaN и inf запрещены;
- при одном модуле `normalized_entropy` определяется как `1.0`;
- при `top_k > module_count` используется `min(top_k, module_count)`;
- `hhi=0` невозможен, но защита от деления на ноль обязательна;
- функция не обращается к модели, GPU или файловой системе.

### 8.7. `finalize_layer_plan`

Сигнатура:

```python
finalize_layer_plan(
    recorder: CalibrationRecorder,
    registry: LayerRegistry,
    *,
    top_k: int,
) -> LayerPlan
```

Смысл:

- заморозить именно финальную проекцию;
- создать канонический handoff для downstream-обучения;
- сохранить полную картину, а не только победителей.

Вход:

- recorder с завершённой калибровкой;
- registry;
- размер Top-K для отчёта.

Выход:

- immutable `LayerPlan`;
- все модули отсортированы;
- Top-K модулей отмечены отдельно;
- итоговые `total_rank` и `selected_module_count`.

Гарантии и ошибки:

- функция обязана вызываться до `_final_lora_rank_update()`;
- отсутствие ровно одной финальной проекции является ошибкой;
- план создаётся из сохранённого `ProjectionEvent`, а не из текущих весов
  оптимизатора после сброса;
- повторный вызов возвращает тот же план и не должен создавать новый run;
- план не зависит от доступности Unsloth.

### 8.8. `validate_layer_plan`

Сигнатура:

```python
validate_layer_plan(
    plan: LayerPlan,
    *,
    model=None,
    adapter_name: str | None = None,
    strict: bool = True,
) -> ValidationReport
```

Смысл:

- не допустить передачи в downstream повреждённого или неоднозначного плана;
- проверить имена, размерности, ранги, alpha и служебные поля.

Вход:

- `LayerPlan`;
- необязательная модель для проверки существования модулей;
- имя адаптера;
- режим strict.

Выход:

- `ValidationReport` со списками errors, warnings и statistics;
- при `strict=True` и наличии errors выбрасывается агрегированная ошибка.

Проверки:

- `schema_version` поддерживается;
- `module_name` уникальны;
- `rank >= 0`;
- `selected == (rank > 0)`;
- `simplex_weight`, `normalized_importance`, `lora_alpha`, `scaling` конечны;
- сумма normalized importance равна `1` с документированным допуском;
- модули разрешаются в модели;
- `in_features`/`out_features` совпадают с моделью;
- ни один обязательный модуль не пропущен;
- нет неэкранированных regex в handoff-представлении.

### 8.9. `verify_applied_plan`

Сигнатура:

```python
verify_applied_plan(
    model,
    plan: LayerPlan,
    *,
    adapter_name: str,
) -> ValidationReport
```

Смысл:

- доказать, что сохранённый план совпадает с фактическими рангами модели;
- поймать расхождение между observer, оптимизатором и PEFT.

Вход:

- модель после `_final_lora_rank_update()`;
- план;
- имя адаптера.

Выход:

- `ValidationReport`.

Гарантии:

- для каждого `rank > 0` проверяется `layer.r[adapter_name]`, размерности A/B и
  `layer.lora_weight[adapter_name] == 1`;
- для `rank = 0` проверяется, что слой не входит в активный набор плана;
- несовпадение является blocking error;
- проверка только читает параметры.

### 8.10. `build_unsloth_handoff`

Сигнатура:

```python
build_unsloth_handoff(
    plan: LayerPlan,
    *,
    rank_floor: int = 1,
    min_weight: float = 0.0,
    top_k: int | None = None,
    alpha_mode: Literal["preserve", "twice_rank"] = "preserve",
) -> UnslothHandoff
```

Смысл:

- преобразовать план важности в обычную LoRA-конфигурацию;
- не импортировать Unsloth и не зависеть от его версии;
- сделать все изменения относительно исходного плана явными.

Вход:

- валидный `LayerPlan`;
- минимальный допустимый ранг;
- порог важности;
- необязательное ограничение Top-K;
- политика alpha.

Выход:

- `target_modules`: анкорированные escaped regex для выбранных модулей;
- `rank_pattern`;
- `alpha_pattern`;
- список исключённых модулей;
- исходный и применённый rank budget;
- оценка trainable parameters;
- warnings о floor, Top-K и изменении scaling.

Правила:

- выбираются только `rank > 0`, `weight >= min_weight` и модули из Top-K, если
  Top-K задан;
- `applied_rank = max(rank, rank_floor)`;
- `alpha_mode="preserve"` сохраняет исходный `lora_alpha`;
- `alpha_mode="twice_rank"` устанавливает `alpha = 2 * applied_rank` и обязан
  добавить warning об изменении scaling;
- `target_modules` не должен использовать значения `rank=0`;
- оценка параметров считается как `sum(applied_rank * (in_features +
  out_features))`;
- handoff не утверждает, что Unsloth его уже принял; приёмка выполняется
  интеграционным тестом.

### 8.11. `render_calibration_report`

Сигнатура:

```python
render_calibration_report(
    *,
    manifest: CalibrationManifest,
    snapshots: Sequence[ProjectionSnapshot],
    plan: LayerPlan,
    handoff: UnslothHandoff | None,
    top_k: int,
) -> str
```

Смысл:

- сформировать Markdown без скрытых вычислений;
- дать человеку ответ на вопросы «что изменилось», «что победило»,
  «насколько результат концентрирован» и «что передаётся дальше».

Вход:

- manifest;
- все снимки проекций;
- финальный план;
- необязательный handoff;
- размер Top-K.

Выход:

- строка валидного Markdown.

Гарантии:

- функция чистая;
- сортировка детерминирована;
- большие массивы не включаются;
- все числа форматируются единообразно;
- пустые значения отображаются как `n/a`;
- отчёт не содержит абсолютных локальных путей;
- при отсутствии handoff раздел Unsloth явно помечается как не сформированный.

### 8.12. `write_calibration_artifacts`

Сигнатура:

```python
write_calibration_artifacts(
    *,
    recorder: CalibrationRecorder,
    plan: LayerPlan,
    handoff: UnslothHandoff | None,
    output_dir: Path,
) -> ArtifactPaths
```

Смысл:

- атомарно сформировать все итоговые артефакты;
- централизовать сериализацию, сортировку и checksums.

Вход:

- recorder;
- финальный план;
- handoff;
- каталог.

Выход:

- `ArtifactPaths` с абсолютными путями и SHA-256;
- manifest в состоянии `completed`, если калибровка завершена.

Гарантии:

- каталог создаётся при необходимости;
- временные файлы находятся в том же каталоге и заменяются атомарно;
- JSON пишется с `ensure_ascii=False`, `allow_nan=False`;
- порядок ключей и массивов стабилен;
- существующие успешные артефакты не перезаписываются без явного `overwrite`;
- при ошибке старые завершённые артефакты не повреждаются;
- SHA-256 вычисляется по итоговому содержимому;
- manifest обновляется последним.

### 8.13. `load_layer_plan`

Сигнатура:

```python
load_layer_plan(path: Path, *, strict: bool = True) -> LayerPlan
```

Смысл:

- безопасно загрузить план в отчётные инструменты или downstream-код;
- обеспечить обратную совместимость схемы.

Вход:

- путь к JSON;
- strict-режим.

Выход:

- `LayerPlan`.

Гарантии:

- JSON читается без исполнения кода;
- проверяется `schema_version`;
- неизвестные поля сохраняются, но не влияют на расчёт;
- отсутствие обязательных полей является ошибкой в strict-режиме;
- выполняется полная структурная валидация до возврата.

### 8.14. `compare_calibration_runs`

Сигнатура:

```python
compare_calibration_runs(
    plans: Sequence[LayerPlan],
    *,
    top_k: int = 20,
) -> StabilityReport
```

Смысл:

- проверить, что распределение не является артефактом одного seed или одной
  калибровочной подвыборки;
- предоставить объективные метрики устойчивости до передачи плана в Unsloth.

Вход:

- два или более валидных плана для одной модели и одной архитектуры;
- Top-K.

Выход:

- `StabilityReport`:
  - union и intersection Top-K;
  - Jaccard Top-K;
  - частота попадания модуля в Top-K;
  - mean/std/min/max ранга;
  - mean/std normalized importance;
  - stable/unstable classification;
  - предупреждение при сравнении разных моделей или адаптеров.

Гарантии:

- модули сопоставляются по каноническому имени;
- несопоставимые размерности являются ошибкой;
- результат сортируется по частоте, затем по среднему рангу, затем по имени;
- отчёт может быть отрендерен в Markdown отдельной функцией
  `render_stability_report`.

### 8.15. `add_calibration_arguments`

Сигнатура:

```python
add_calibration_arguments(
    parser: argparse.ArgumentParser,
) -> None
```

Смысл:

- единообразно подключить параметры калибровки в CLI экспериментов;
- отделить парсинг и валидацию от бизнес-логики.

Вход:

- существующий `ArgumentParser`.

Выход:

- parser с новыми аргументами.

Минимальные аргументы:

- `--calibration-output-dir`;
- `--calibration-top-k`, default `20`;
- `--calibration-only`;
- `--calibration-no-artifacts`;
- `--calibration-rank-floor`, default `1`;
- `--calibration-alpha-mode`, choices `preserve` и `twice_rank`;
- `--calibration-min-weight`, default `0.0`;
- `--calibration-unsloth-top-k`;
- `--calibration-overwrite`.

### 8.16. `validate_calibration_arguments`

Сигнатура:

```python
validate_calibration_arguments(args: argparse.Namespace) -> None
```

Смысл:

- поймать несовместимые комбинации до загрузки модели и начала обучения.

Вход:

- parsed arguments.

Выход:

- `None`, если конфигурация валидна;
- parser-like error или `ValueError` с понятным сообщением иначе.

Проверки:

- `fat_step > 0`;
- `max_fat_steps > 0` при включённой калибровке;
- `top_k > 0`;
- `rank_floor >= 1`;
- `0 <= min_weight <= 1`;
- `--calibration-only` несовместим с явным `max_steps`, не равным
  `fat_step * max_fat_steps`;
- `--calibration-only` требует включённые артефакты;
- `--calibration-no-artifacts` несовместим с `--calibration-only`;
- output directory не совпадает с исходными файлами проекта.

### 8.17. `create_calibration_session`

Сигнатура:

```python
create_calibration_session(
    *,
    model,
    optimizer,
    args,
    run_metadata,
) -> CalibrationSession
```

Смысл:

- связать registry, recorder, optimizer observer и Trainer callback;
- не дублировать wiring в каждом experiment CLI.

Вход:

- модель;
- `SimplexAdamW`;
- CLI arguments;
- метаданные запуска.

Выход:

- `CalibrationSession` с recorder и методом `finalize()`.

Гарантии:

- повторное подключение observer к тому же optimizer запрещено;
- при `--calibration-no-artifacts` observer не создаётся;
- session валидирует, что optimizer имеет group `weight_params`;
- session не загружает данные и не выбирает устройство;
- при исключении в training loop `close(failed=True)` сохраняет partial manifest.

### 8.18. `render_console_summary`

Сигнатура:

```python
render_console_summary(
    plan: LayerPlan,
    *,
    top_k: int = 10,
) -> str
```

Смысл:

- дать краткий результат сразу после прогона без чтения JSON и Markdown.

Вход:

- финальный план;
- Top-K.

Выход:

- короткий текст:
  - завершена ли калибровка;
  - число активных модулей;
  - суммарный rank budget;
  - Top-K строк `module`, `weight`, `rank`;
  - пути к полному отчёту и handoff;
  - предупреждение, если распределение сильно концентрировано.

Гарантии:

- stdout не содержит полную матрицу;
- вывод не зависит от цветового терминала;
- при незавершённой калибровке возвращается явное сообщение, а не фиктивный
  Top-K.

## 9. Изменения в существующих компонентах

### 9.1. `SimplexAdamW`

- сохранить математику текущей реализации;
- добавить необязательный `calibration_observer`;
- в начале `step()` рассчитать и передать observer-у нормы LoRA- и weight-
  градиентов;
- добавить метод `add_microbatch_loss(loss)`;
- агрегировать loss по optimizer step;
- при каждой проекции передать точные локальные
  `w_vector_old`, `w_vector`, `new_lora_ranks_all`, selected indices и
  `is_final_projection`;
- после вызова observer-а продолжить существующий rank update;
- после финального rank update вызвать `verify_applied_plan`;
- не добавлять I/O или JSON-сериализацию непосредственно в математическую
  ветку оптимизатора.

### 9.2. PEFT-плагин

- предоставить read-only helper для получения текущего ранга, alpha и scaling;
- не менять формат state dict без отдельного migration plan;
- `layer_plan.json` не должен зависеть от наличия `lora_weight=1` после reset;
- обеспечить канонические имена модулей независимо от префиксов
  `base_model.model`;
- добавить тест сохранения плана до `_final_lora_rank_update()`.

### 9.3. `decompile.py`

- использовать `SimplexCalibrationTrainer` вместо stock `Trainer` для точного
  step loss;
- создать `CalibrationSession` перед `trainer.train()`;
- в `training_summary.json` добавить:
  - путь к `calibration_report.md`;
  - путь к `layer_plan.json`;
  - путь к `unsloth_handoff.json`;
  - `calibration_completed`;
  - `selected_module_count`;
  - `total_rank`;
  - финальные метрики концентрации.
- при `--calibration-only` остановиться ровно после
  `fat_step * max_fat_steps` optimizer steps;
- в calibration-only не сохранять бессмысленный обученный адаптер по умолчанию;
- не выполнять evaluation качества, если оно не входит в отдельный сценарий.

### 9.4. Другие эксперименты

Общий слой `calibration` должен быть совместим с другими CLI, использующими
`SimplexAdamW`. Подключение в `glue`, `nlg`, `squad` и `llm` выполняется через
`create_calibration_session`; эксперимент-specific код не должен
самостоятельно формировать JSON.

## 10. Режимы запуска

### 10.1. Полная калибровка и обучение

Калибровочные артефакты создаются всегда, затем обучение продолжается до
`num_train_epochs` или `max_steps`. `steps.jsonl` пишется только до финальной
проекции, чтобы не раздувать файл на тысячи downstream-шагов.

### 10.2. Только калибровка

`--calibration-only`:

- вычисляет `required_steps = fat_step * max_fat_steps`;
- требует, чтобы это значение было положительным;
- запускает ровно это число optimizer steps;
- сохраняет план до финального reset;
- выполняет финальный rank update для проверки применяемости плана;
- сохраняет handoff;
- останавливает Trainer;
- не выдаёт финальный адаптер за обученную модель.

### 10.3. Наблюдение отключено

`--calibration-no-artifacts`:

- не создаёт recorder и JSONL;
- не меняет математику `SimplexAdamW`;
- сохраняет прежние stdout-сообщения только при явно включённом legacy logging;
- несовместим с `--calibration-only`.

## 11. Обработка ошибок и частичных результатов

Сценарии:

- **Ошибка до первой проекции.** Manifest: `failed`, plan отсутствует,
  `calibration_completed=false`.
- **Ошибка между проекциями.** Manifest: `partial`; завершённые JSONL-строки
  сохраняются, plan отсутствует.
- **Ошибка записи финального плана.** Обучение останавливается до reset A/B,
  чтобы диагностическое состояние не потерялось.
- **Ошибка verify после reset.** Manifest: `failed`; план сохраняется с
  `verification.error`, downstream handoff помечается непригодным.
- **Прерывание сигналом.** Обработчик завершения выполняет flush и помечает
  запуск `partial`, если это возможно без маскировки исходного сигнала.
- **Повторный запуск в тот же каталог.** Без `overwrite` создаётся ошибка;
  автоинкремент каталога не допускается.

## 12. Требования к метрикам и интерпретации

Отчёт обязан явно разделять:

- изученный simplex weight;
- нормализованную importance;
- фактически выделенный rank;
- scaling и alpha;
- значение loss.

Нельзя называть норму градиента важностью слоя. `weight_grad_norm` — только
диагностическая величина оптимизации.

Отчёт должен предупреждать, если:

- доля одного модуля превышает заданный порог, default `0.5` normalized
  importance;
- `effective_module_count < 2`;
- более 90% модулей имеют нулевой rank;
- финальный ранг одного модуля превышает `4 * default_lora_rank`;
- распределение между Top-K почти не меняется и выглядит winner-take-all;
- `loss` недоступен;
- run остановлен до финальной проекции;
- handoff изменил rank из-за `rank_floor` или Top-K.

Пороговые значения должны быть параметрами отчёта, а не магическими числами в
рендерере.

## 13. Производительность

- observer не должен выполнять синхронизацию GPU на каждом micro-batch;
- loss допускается накапливать на device до optimizer step;
- на проекции допустим один перенос per-layer массивов на CPU;
- JSONL flush выполняется на границе optimizer step, максимум 50 раз для
  типового калибровочного сценария;
- полные веса адаптеров в артефакты не сохраняются;
- overhead observer в не-проекционных шагах не должен превышать 2% времени
  обучения; это проверяется benchmark-тестом на фиксированной модели;
- финальная генерация отчёта может выполняться синхронно.

## 14. Безопасность и целостность

- `layer_plan.json` и handoff проверяются как данные, без `eval`, pickle и
  импорта кода;
- regex-шаблоны строятся через `re.escape`;
- output directory не должен быть символической ссылкой наружу проекта без
  явного разрешения;
- запись не должна следовать по небезопасным symlink;
- `run_id` и relative paths проходят allowlist-сериализацию;
- все созданные файлы имеют явно документированные имена;
- SHA-256 фиксирует содержимое каждого завершённого артефакта.

## 15. Тестирование и критерии приёмки

### 15.1. Unit-тесты без GPU

- `compute_calibration_metrics` на известных векторах:
  uniform, один победитель, несколько ненулевых компонентов;
- `build_layer_registry` на tiny PEFT-модели;
- уникальность и сортировка registry;
- ровно один `record_step` на optimizer step;
- ровно один `record_projection` на projection;
- сохранение `ProjectionEvent` до финального reset;
- `rank = ceil(weight * default_rank)` во всех событиях;
- JSON не содержит NaN/Infinity;
- детерминированный порядок CSV и JSON;
- atomic write сохраняет предыдущий файл при инъекции ошибки;
- strict loader отклоняет повреждённый schema;
- `validate_layer_plan` ловит дубли и несовпадение dimensions;
- `verify_applied_plan` ловит подменённый rank;
- handoff не включает rank 0;
- `rank_floor` и `top_k` отражаются в warnings;
- `compare_calibration_runs` корректно считает Jaccard и частоты.

### 15.2. Интеграционные тесты на CPU

Использовать tiny causal LM и существующий CPU-сценарий PEFT-плагина:

- запустить 2 цикла калибровки;
- убедиться, что создан полный набор артефактов;
- проверить, что финальный `layer_plan.json` соответствует модели;
- убедиться, что `_final_lora_rank_update()` не меняет уже сохранённый план;
- проверить, что обычное обучение без artifacts даёт тот же loss trajectory,
  что и до рефакторинга;
- проверить запуск с artifacts: численная траектория не меняется в пределах
  отсутствия наблюдаемых side effects;
- проверить `--calibration-only` и точное число optimizer steps.

### 15.3. GPU-тесты, opt-in

- NF4 forward/backward с observer;
- корректность loss при gradient accumulation;
- отсутствие лишней синхронизации на каждом micro-batch;
- сохранение событий при финальной проекции;
- ROCm и CUDA дают одинаковую размерность и структуру артефактов; численные
  значения могут отличаться в допустимом tolerance;
- benchmark overhead не превышает установленный порог.

### 15.4. Критерии готовности

ТЗ считается реализованным, когда одновременно выполнены условия:

1. Стандартный `simplexlora-decompile` создаёт manifest, step history,
   projection history, CSV, Markdown, layer plan и handoff.
2. `global_step` финальной проекции совпадает с ожидаемым циклом
   `fat_step * max_fat_steps`.
3. Для Qwen-подобной модели в плане присутствуют все target modules, включая
   нулевые ранги.
4. `layer_plan.json` проходит `validate_layer_plan` и
   `verify_applied_plan`.
5. Top-K в Markdown и JSON совпадает.
6. `--calibration-only` останавливается на последней проекции и не выдаёт
   reset-адаптер за результат обучения.
7. Включение observer не изменяет оптимизационную математику.
8. Повреждённый или неполный план не может быть превращён в handoff.
9. Повторный запуск в тот же каталог не смешивает артефакты.
10. Два запуска с разными seed сравниваются через `compare_calibration_runs`.
11. `uv check --no-project`, `uv run --extra cuda ruff check` и
    `uv run --extra cuda ruff format --check` проходят; на AMD
    `--extra cuda` заменяется на `--extra rocm`.
12. CPU unit- и integration-тесты проходят без GPU.

## 16. Рекомендуемая декомпозиция кода

```text
src/simplexlora/calibration/
├── __init__.py
├── models.py          # dataclasses, schema models и enums
├── registry.py        # build_layer_registry
├── metrics.py         # compute_calibration_metrics
├── recorder.py        # CalibrationRecorder, session
├── plan.py            # finalize, validate, load, verify
├── handoff.py         # build_unsloth_handoff
├── report.py          # Markdown и console rendering
└── compare.py         # сравнение нескольких запусков
```

CLI-интеграция должна оставаться тонкой и вызывать публичные функции пакета.

## 17. Порядок реализации

1. Registry, data models и точные ProjectionEvent.
2. Observer в `SimplexAdamW` без изменения математики.
3. Trainer loss capture и `CalibrationRecorder`.
4. Финальный `LayerPlan` до reset и `verify_applied_plan`.
5. JSONL, JSON и CSV.
6. Markdown-отчёт и console summary.
7. `--calibration-only` и интеграция с `decompile.py`.
8. Unsloth handoff без зависимости от Unsloth.
9. `compare_calibration_runs`.
10. GPU benchmark и opt-in regression tests.

После шага 4 калибровка уже должна быть пригодна для ручного анализа. После шага
8 должен существовать полный handoff-контракт для downstream-обучения.
