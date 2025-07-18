# Makefile для тестирования производительности матричного умножения
# Автор: Для экспериментов с CUDA
# Дата: Июнь 2025

# Компилятор и флаги
NVCC = nvcc
CFLAGS = -std=c++14
TARGET = matrix.out
SOURCE = matrixMul.cu

# Размеры матриц для тестирования
SIZES = 640 6400

# Размеры блоков для тестирования
BLOCK_SIZES = 16 32

# Количество элементов на поток для тестирования
ELEMENTS_PER_THREAD_VALUES = 1 2 4 8

# Текущие значения (по умолчанию)
CURRENT_BLOCK_SIZE = 32
CURRENT_ELEMENTS = 4

# Цвета для красивого вывода
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
PURPLE = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

.PHONY: all clean test-all test-size help table test-elements test-float4 test-matrix test-blocks test-combinations mega-test

# Основная цель - компиляция
all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "$(CYAN)🔨 Компилируем CUDA kernel...$(NC)"
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)
	@echo "$(GREEN)✅ Компиляция завершена!$(NC)"

# Тестирование всех размеров
test-all: $(TARGET)
	@echo "$(PURPLE)🚀 НАЧИНАЕМ ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ 🚀$(NC)"
	@echo "$(YELLOW)================================================$(NC)"
	@echo "$(BLUE)| Размер  | Время (мс) | GFlop/s | Элементы/поток |$(NC)"
	@echo "$(YELLOW)================================================$(NC)"
	@for size in $(SIZES); do \
		echo "$(CYAN)🧪 Тестируем размер: $${size}x$${size}$(NC)"; \
		./$(TARGET) -wA=$$size -hA=$$size -wB=$$size -hB=$$size | \
		grep -E "(ЭЛЕМЕНТОВ НА ПОТОК|Performance)" | \
		awk 'BEGIN{elements=0; perf=""} \
		/ЭЛЕМЕНТОВ НА ПОТОК/ {elements=$$4} \
		/Performance/ {split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]} \
		END{printf "$(GREEN)| %6s  | %10s | %7s | %14s |$(NC)\n", "'$$size'", time, perf, elements}'; \
	done
	@echo "$(YELLOW)================================================$(NC)"
	@echo "$(GREEN)✅ Тестирование завершено!$(NC)"

# Тестирование конкретного размера
test-size: $(TARGET)
	@if [ -z "$(SIZE)" ]; then \
		echo "$(RED)❌ Ошибка: Укажите размер с SIZE=число$(NC)"; \
		echo "$(YELLOW)Пример: make test-size SIZE=800$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)🧪 Тестируем размер: $(SIZE)x$(SIZE)$(NC)"
	./$(TARGET) -wA=$(SIZE) -hA=$(SIZE) -wB=$(SIZE) -hB=$(SIZE)

# Создание красивой таблицы для копирования
table: $(TARGET)
	@echo "$(PURPLE)📊 СОЗДАЕМ ТАБЛИЦУ ДЛЯ ОТЧЕТА 📊$(NC)"
	@echo "$(YELLOW)==========================================$(NC)"
	@echo "Размер матрицы | Время (мс) | Производительность (GFlop/s) | Элементы на поток"
	@echo "---------------|------------|------------------------------|------------------"
	@for size in $(SIZES); do \
		./$(TARGET) -wA=$$size -hA=$$size -wB=$$size -hB=$$size 2>/dev/null | \
		grep -E "(ЭЛЕМЕНТОВ НА ПОТОК|Performance)" | \
		awk 'BEGIN{elements=0; perf=""; time=""} \
		/ЭЛЕМЕНТОВ НА ПОТОК/ {elements=$$4} \
		/Performance/ {split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]} \
		END{printf "%13sx%-2s | %10s | %28s | %16s\n", "'$$size'", "'$$size'", time, perf, elements}'; \
	done
	@echo "$(YELLOW)==========================================$(NC)"

# Быстрый тест с разными ELEMENTS_PER_THREAD
test-elements: $(TARGET)
	@echo "$(PURPLE)🔬 ТЕСТИРУЕМ РАЗНЫЕ ЭЛЕМЕНТЫ НА ПОТОК 🔬$(NC)"
	@echo "$(YELLOW)Размер матрицы: 800x800$(NC)"
	@echo "$(BLUE)Элементы/поток | Производительность (GFlop/s) | Время (мс)$(NC)"
	@echo "$(YELLOW)----------------------------------------------------$(NC)"
	@for elements in 1 2 4 8; do \
		echo "$(CYAN)🧪 Тестируем $$elements элементов на поток$(NC)"; \
		./$(TARGET) -wA=800 -hA=800 -wB=800 -hB=800 -elements_per_thread=$$elements 2>/dev/null | \
		grep "Performance" | \
		awk '{split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]; printf "$(GREEN)%13s  | %26s | %10s$(NC)\n", "'$$elements'", perf, time}'; \
	done
	@echo "$(YELLOW)----------------------------------------------------$(NC)"

# Тест с включенной float4 векторизацией
test-float4: $(TARGET)
	@echo "$(PURPLE)🌟 ТЕСТИРУЕМ FLOAT4 ВЕКТОРИЗАЦИЮ 🌟$(NC)"
	@echo "$(YELLOW)Размер матрицы: 800x800, 4 элемента на поток$(NC)"
	@sed -i "s/#define USE_FLOAT4.*/#define USE_FLOAT4 1/" $(SOURCE)
	@sed -i "s/#define ELEMENTS_PER_THREAD.*/#define ELEMENTS_PER_THREAD 4/" $(SOURCE)
	@$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)
	@echo "$(CYAN)🚀 Float4 векторизация ВКЛЮЧЕНА$(NC)"
	@./$(TARGET) -wA=800 -hA=800 -wB=800 -hB=800
	@sed -i "s/#define USE_FLOAT4.*/#define USE_FLOAT4 0/" $(SOURCE)
	@echo "$(GREEN)✅ Тест float4 завершен (векторизация отключена)$(NC)"

# МЕГА ТЕСТ: Все комбинации блоков и элементов для каждого размера матрицы
mega-test: $(TARGET)
	@echo "$(PURPLE)🚀🚀🚀 МЕГА ТЕСТ ВСЕХ КОМБИНАЦИЙ 🚀🚀🚀$(NC)"
	@echo "$(YELLOW)===============================================$(NC)"
	@for size in $(SIZES); do \
		echo "$(CYAN)🎯 РАЗМЕР МАТРИЦЫ: $${size}x$${size}$(NC)"; \
		echo "$(BLUE)Блок | Элементы/поток | Время (мс) | GFlop/s$(NC)"; \
		echo "$(YELLOW)-----|----------------|------------|--------$(NC)"; \
		for block in $(BLOCK_SIZES); do \
			for elements in $(ELEMENTS_PER_THREAD_VALUES); do \
				echo "$(CYAN)🔧 Тестируем: блок=$$block, элементы=$$elements$(NC)"; \
				perf_data=$$(./$(TARGET) -wA=$$size -hA=$$size -wB=$$size -hB=$$size -blocksize=$$block -elements_per_thread=$$elements 2>/dev/null | grep "Performance"); \
				if [ ! -z "$$perf_data" ]; then \
					echo "$$perf_data" | awk '{split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]; printf "$(GREEN)%4s | %14s | %10s | %7s$(NC)\n", "'$$block'", "'$$elements'", time, perf}'; \
				fi; \
			done; \
		done; \
		echo "$(YELLOW)-----|----------------|------------|--------$(NC)"; \
		echo ""; \
	done
	@echo "$(GREEN)✅ МЕГА ТЕСТ ЗАВЕРШЕН!$(NC)"

# Тест всех комбинаций для одного размера матрицы (800x800)
test-combinations: $(TARGET)
	@echo "$(PURPLE)🧪 ТЕСТИРУЕМ ВСЕ КОМБИНАЦИИ для 800x800 🧪$(NC)"
	@echo "$(YELLOW)================================================$(NC)"
	@echo "$(BLUE)Размер блока | Элементы/поток | Время (мс) | Производительность (GFlop/s)$(NC)"
	@echo "$(YELLOW)-------------|----------------|------------|-----------------------------$(NC)"
	@for block in $(BLOCK_SIZES); do \
		for elements in $(ELEMENTS_PER_THREAD_VALUES); do \
			echo "$(CYAN)🔧 Тестируем: блок=$$block, элементы=$$elements$(NC)"; \
			./$(TARGET) -wA=800 -hA=800 -wB=800 -hB=800 -blocksize=$$block -elements_per_thread=$$elements 2>/dev/null | \
			grep "Performance" | \
			awk '{split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]; printf "$(GREEN)%12s | %14s | %10s | %27s$(NC)\n", "'$$block'", "'$$elements'", time, perf}'; \
		done; \
	done
	@echo "$(YELLOW)-------------|----------------|------------|-----------------------------$(NC)"
	@echo "$(GREEN)✅ Тест комбинаций завершен!$(NC)"

# Отдельная таблица для размеров блоков
test-blocks: $(TARGET)
	@echo "$(PURPLE)🔧 АНАЛИЗ ВЛИЯНИЯ РАЗМЕРА БЛОКА 🔧$(NC)"
	@echo "$(YELLOW)Фиксированные параметры: элементы=4, размер=800x800$(NC)"
	@echo "$(BLUE)Размер блока | Время (мс) | Производительность (GFlop/s)$(NC)"
	@echo "$(YELLOW)-------------|------------|-----------------------------$(NC)"
	@for block in $(BLOCK_SIZES); do \
		echo "$(CYAN)🧪 Тестируем блок размером: $$block$(NC)"; \
		./$(TARGET) -wA=800 -hA=800 -wB=800 -hB=800 -blocksize=$$block -elements_per_thread=4 2>/dev/null | \
		grep "Performance" | \
		awk '{split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]; printf "$(GREEN)%12s | %10s | %27s$(NC)\n", "'$$block'", time, perf}'; \
	done
	@echo "$(YELLOW)-------------|------------|-----------------------------$(NC)"

# Создание CSV файла для импорта в Excel/Google Sheets
csv-export: $(TARGET)
	@echo "$(PURPLE)📊 ЭКСПОРТ В CSV ФОРМАТ 📊$(NC)"
	@echo "Matrix_Size,Block_Size,Elements_Per_Thread,Time_ms,Performance_GFlops" > performance_results.csv
	@for size in $(SIZES); do \
		for block in $(BLOCK_SIZES); do \
			for elements in $(ELEMENTS_PER_THREAD_VALUES); do \
				perf_data=$$(./$(TARGET) -wA=$$size -hA=$$size -wB=$$size -hB=$$size -blocksize=$$block -elements_per_thread=$$elements 2>/dev/null | grep "Performance"); \
				if [ ! -z "$$perf_data" ]; then \
					echo "$$perf_data" | awk '{split($$0,a,"="); split(a[2],b,","); perf=b[1]; split(a[3],c,","); time=c[1]; printf "'$$size','$$block','$$elements',%s,%s\n", time, perf}' >> performance_results.csv; \
				fi; \
			done; \
		done; \
	done
	@echo "$(GREEN)✅ Результаты сохранены в performance_results.csv$(NC)"

# Поиск лучшей конфигурации
find-best: $(TARGET)
	@echo "$(PURPLE)🏆 ПОИСК ЛУЧШЕЙ КОНФИГУРАЦИИ 🏆$(NC)"
	@echo "$(YELLOW)Тестируем размер 800x800...$(NC)"
	@best_perf=0; best_config=""; \
	for block in $(BLOCK_SIZES); do \
		for elements in $(ELEMENTS_PER_THREAD_VALUES); do \
			perf=$$(./$(TARGET) -wA=800 -hA=800 -wB=800 -hB=800 -blocksize=$$block -elements_per_thread=$$elements 2>/dev/null | grep "Performance" | awk '{split($$0,a,"="); split(a[2],b,","); print b[1]}' | tr -d ' '); \
			if [ ! -z "$$perf" ]; then \
				if (( $$(echo "$$perf > $$best_perf" | bc -l) )); then \
					best_perf=$$perf; \
					best_config="блок=$$block, элементы=$$elements"; \
				fi; \
				echo "$(CYAN)Блок $$block, элементы $$elements: $$perf GFlop/s$(NC)"; \
			fi; \
		done; \
	done; \
	echo "$(GREEN)🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ: $$best_config с $$best_perf GFlop/s$(NC)"

# Очистка
clean:
	@echo "$(YELLOW)🧹 Очищаем временные файлы...$(NC)"
	rm -f $(TARGET)
	@echo "$(GREEN)✅ Очистка завершена!$(NC)"

# Помощь
help:
	@echo "$(PURPLE)📖 ПОМОЩЬ ПО MAKEFILE 📖$(NC)"
	@echo "$(YELLOW)==============================$(NC)"
	@echo "$(GREEN)🔨 КОМПИЛЯЦИЯ:$(NC)"
	@echo "$(CYAN)make all$(NC)               - Компилировать программу"
	@echo ""
	@echo "$(GREEN)🧪 ОСНОВНЫЕ ТЕСТЫ:$(NC)"
	@echo "$(CYAN)make test-all$(NC)          - Тестировать все размеры матриц"
	@echo "$(CYAN)make test-size SIZE=N$(NC)   - Тестировать конкретный размер"
	@echo "$(CYAN)make table$(NC)             - Создать таблицу для отчета"
	@echo ""
	@echo "$(GREEN)🔬 СПЕЦИАЛЬНЫЕ ТЕСТЫ:$(NC)"
	@echo "$(CYAN)make test-elements$(NC)     - Тест разных элементов на поток"
	@echo "$(CYAN)make test-blocks$(NC)       - Анализ влияния размера блока"
	@echo "$(CYAN)make test-combinations$(NC) - Все комбинации для 800x800"
	@echo "$(CYAN)make test-float4$(NC)       - Тест float4 векторизации"
	@echo ""
	@echo "$(GREEN)🚀 МЕГА ТЕСТЫ:$(NC)"
	@echo "$(CYAN)make mega-test$(NC)         - ВСЕ комбинации для ВСЕХ размеров"
	@echo "$(CYAN)make find-best$(NC)         - Найти лучшую конфигурацию"
	@echo "$(CYAN)make csv-export$(NC)        - Экспорт результатов в CSV"
	@echo ""
	@echo "$(GREEN)🧹 УТИЛИТЫ:$(NC)"
	@echo "$(CYAN)make clean$(NC)             - Очистить временные файлы"
	@echo "$(CYAN)make help$(NC)              - Показать эту справку"
	@echo "$(YELLOW)==============================$(NC)"
	@echo "$(GREEN)💡 Примеры использования:$(NC)"
	@echo "  make test-size SIZE=1600"
	@echo "  make table > results.txt"
	@echo "  make csv-export"
	@echo "  make find-best"
	@echo "$(YELLOW)==============================$(NC)"
	@echo "$(PURPLE)🎯 Размеры матриц: $(SIZES)$(NC)"
	@echo "$(PURPLE)🔧 Размеры блоков: $(BLOCK_SIZES)$(NC)"
	@echo "$(PURPLE)⚡ Элементы/поток: $(ELEMENTS_PER_THREAD_VALUES)$(NC)"
