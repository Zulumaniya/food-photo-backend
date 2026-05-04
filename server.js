import "dotenv/config";

import express from "express";
import multer from "multer";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const PORT = Number(process.env.PORT || 8080);
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 45_000);
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const DEFAULT_WARNING = "Порции определены приблизительно. Проверьте значения перед добавлением.";
const DEFAULT_NOTE = "Оценка по фото";

const FoodPhotoItemSchema = z.object({
  name: z.string().nullable(),
  grams: z.number().nullable(),
  calories: z.number().nullable(),
  proteins: z.number().nullable(),
  fats: z.number().nullable(),
  carbs: z.number().nullable(),
  confidence: z.number().nullable(),
  note: z.string().nullable()
});

const FoodPhotoResponseSchema = z.object({
  items: z.array(FoodPhotoItemSchema),
  warnings: z.array(z.string())
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_IMAGE_BYTES,
    files: 1
  },
  fileFilter: (_req, file, cb) => {
    if (!SUPPORTED_IMAGE_TYPES.has(file.mimetype)) {
      return cb(new UnsupportedMediaTypeError("Only JPEG, PNG and WebP images are supported"));
    }
    cb(null, true);
  }
});

const app = express();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: OPENAI_TIMEOUT_MS
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    service: "food-photo-backend"
  });
});

app.post("/food-photo/analyze", upload.single("image"), async (req, res) => {
  if (!process.env.OPENAI_API_KEY) {
    return sendAndroidError(res, 500, "Сервис анализа временно недоступен.");
  }

  if (!req.file) {
    return sendAndroidError(res, 400, "Не передано изображение для анализа.");
  }

  const locale = normalizeText(req.body?.locale) || "ru";
  const mealType = normalizeText(req.body?.mealType) || "unknown";

  try {
    const result = await analyzeFoodPhoto({
      imageBuffer: req.file.buffer,
      mimeType: req.file.mimetype,
      locale,
      mealType
    });

    res.json(result);
  } catch (error) {
    console.error("[food-photo-backend] analyze failed", safeError(error));
    sendAndroidError(res, 502, "Не удалось проанализировать фото. Попробуйте позже.");
  }
});

app.use((error, _req, res, _next) => {
  if (error instanceof UnsupportedMediaTypeError) {
    return sendAndroidError(res, 415, "Поддерживаются только изображения JPEG, PNG или WebP.");
  }

  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return sendAndroidError(res, 413, "Изображение слишком большое. Максимальный размер: 10 MB.");
  }

  console.error("[food-photo-backend] request failed", safeError(error));
  sendAndroidError(res, 500, "Не удалось обработать запрос.");
});

async function analyzeFoodPhoto({ imageBuffer, mimeType, locale, mealType }) {
  const imageBase64 = imageBuffer.toString("base64");
  const imageUrl = `data:${mimeType};base64,${imageBase64}`;

  const response = await openai.responses.parse(
    {
      model: OPENAI_MODEL,
      input: [
        {
          role: "system",
          content: buildSystemPrompt()
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: [
                `locale: ${locale}`,
                `mealType: ${mealType}`,
                "Проанализируй фото еды и верни только структурированный JSON по схеме."
              ].join("\n")
            },
            {
              type: "input_image",
              image_url: imageUrl
            }
          ]
        }
      ],
      text: {
        format: zodTextFormat(FoodPhotoResponseSchema, "food_photo_analysis")
      }
    },
    {
      timeout: OPENAI_TIMEOUT_MS
    }
  );

  if (!response.output_parsed) {
    throw new Error("OpenAI returned empty parsed output");
  }

  const parsed = FoodPhotoResponseSchema.parse(response.output_parsed);
  return normalizeAnalysis(parsed);
}

function buildSystemPrompt() {
  return [
    "Ты анализируешь фото еды для дневника питания.",
    "Определи только видимые блюда и продукты.",
    "Оцени массу порции в граммах.",
    "Оцени калории, белки, жиры и углеводы.",
    "Ответ должен быть на русском языке.",
    "Верни результат только в JSON по заданной схеме.",
    "Не добавляй продукты, которых не видно.",
    "Если блюдо сложное, можно вернуть его одним item, например \"Борщ\", а не раскладывать на все ингредиенты.",
    "Масса и КБЖУ приблизительные.",
    "Если порция неочевидна, оцени приблизительно, снизь confidence и добавь warning.",
    "Если еда не распознана, верни пустой массив items и warning."
  ].join("\n");
}

function normalizeAnalysis(raw) {
  const items = raw.items
    .map(normalizeItem)
    .filter(Boolean);

  const warnings = raw.warnings
    .map(normalizeText)
    .filter(Boolean);

  if (items.length === 0 && warnings.length === 0) {
    warnings.push("Не удалось уверенно распознать продукты на фото.");
  }

  if (items.length > 0 && warnings.length === 0) {
    warnings.push(DEFAULT_WARNING);
  }

  return {
    items,
    warnings
  };
}

function normalizeItem(item) {
  const name = normalizeText(item.name);
  const grams = positiveNumber(item.grams);

  if (!name || grams <= 0) {
    return null;
  }

  return {
    name,
    grams,
    calories: nonNegativeNumber(item.calories),
    proteins: nonNegativeNumber(item.proteins),
    fats: nonNegativeNumber(item.fats),
    carbs: nonNegativeNumber(item.carbs),
    confidence: clamp01(item.confidence),
    note: normalizeText(item.note) || DEFAULT_NOTE
  };
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? round1(number) : 0;
}

function nonNegativeNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? round1(number) : 0;
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.min(1, Math.max(0, round2(number)));
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function round2(value) {
  return Math.round(value * 100) / 100;
}

function normalizeText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function sendAndroidError(res, status, warning) {
  return res.status(status).json({
    items: [],
    warnings: [warning]
  });
}

function safeError(error) {
  return {
    name: error?.name,
    message: error?.message,
    status: error?.status,
    code: error?.code
  };
}

class UnsupportedMediaTypeError extends Error {}

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[food-photo-backend] listening on http://0.0.0.0:${PORT}`);
});
