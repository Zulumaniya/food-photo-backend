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
const RESPONSE_LANGUAGE_RUSSIAN = "Russian";
const RESPONSE_LANGUAGE_ENGLISH = "English";
const LOCALIZED_TEXT = {
  [RESPONSE_LANGUAGE_RUSSIAN]: {
    defaultWarning: "Порции определены приблизительно. Проверьте значения перед добавлением.",
    defaultNote: "Оценка по фото",
    emptyRecognitionWarning: "Не удалось уверенно распознать продукты на фото.",
    serviceUnavailable: "Сервис анализа временно недоступен.",
    missingImage: "Не передано изображение для анализа.",
    analyzeFailed: "Не удалось проанализировать фото. Попробуйте позже.",
    unsupportedMediaType: "Поддерживаются только изображения JPEG, PNG или WebP.",
    fileTooLarge: "Изображение слишком большое. Максимальный размер: 10 MB.",
    requestFailed: "Не удалось обработать запрос."
  },
  [RESPONSE_LANGUAGE_ENGLISH]: {
    defaultWarning: "Portions are estimated. Please review the values before adding them.",
    defaultNote: "Photo estimate",
    emptyRecognitionWarning: "Could not confidently recognize products in the photo.",
    serviceUnavailable: "The analysis service is temporarily unavailable.",
    missingImage: "No image was provided for analysis.",
    analyzeFailed: "Could not analyze the photo. Please try again later.",
    unsupportedMediaType: "Only JPEG, PNG, or WebP images are supported.",
    fileTooLarge: "The image is too large. Maximum size: 10 MB.",
    requestFailed: "Could not process the request."
  }
};

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
  const receivedLanguageCode = normalizeText(req.body?.languageCode);
  const receivedLocale = normalizeText(req.body?.locale);
  const requestedLanguageCode = resolveRequestedLanguageCode(req.body);
  const responseLanguageName = resolveResponseLanguageName(requestedLanguageCode);
  const text = localizedText(responseLanguageName);

  console.log("[food-photo-backend] language", {
    receivedLanguageCode,
    receivedLocale,
    requestedLanguageCode,
    responseLanguageName
  });

  if (!process.env.OPENAI_API_KEY) {
    return sendAndroidError(res, 500, text.serviceUnavailable);
  }

  if (!req.file) {
    return sendAndroidError(res, 400, text.missingImage);
  }

  const mealType = normalizeText(req.body?.mealType) || "unknown";

  try {
    const result = await analyzeFoodPhoto({
      imageBuffer: req.file.buffer,
      mimeType: req.file.mimetype,
      requestedLanguageCode,
      responseLanguageName,
      mealType
    });

    res.json(result);
  } catch (error) {
    console.error("[food-photo-backend] analyze failed", safeError(error));
    sendAndroidError(res, 502, text.analyzeFailed);
  }
});

app.use((error, req, res, _next) => {
  const responseLanguageName = resolveResponseLanguageName(resolveRequestedLanguageCode(req.body));
  const text = localizedText(responseLanguageName);

  if (error instanceof UnsupportedMediaTypeError) {
    return sendAndroidError(res, 415, text.unsupportedMediaType);
  }

  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return sendAndroidError(res, 413, text.fileTooLarge);
  }

  console.error("[food-photo-backend] request failed", safeError(error));
  sendAndroidError(res, 500, text.requestFailed);
});

async function analyzeFoodPhoto({ imageBuffer, mimeType, requestedLanguageCode, responseLanguageName, mealType }) {
  const imageBase64 = imageBuffer.toString("base64");
  const imageUrl = `data:${mimeType};base64,${imageBase64}`;

  const response = await openai.responses.parse(
    {
      model: OPENAI_MODEL,
      input: [
        {
          role: "system",
          content: buildSystemPrompt(responseLanguageName)
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: [
                `requestedLanguageCode: ${requestedLanguageCode}`,
                `responseLanguageName: ${responseLanguageName}`,
                `mealType: ${mealType}`,
                "Analyze the food photo and return only structured JSON matching the schema."
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
  return normalizeAnalysis(parsed, responseLanguageName);
}

function buildSystemPrompt(responseLanguageName) {
  return [
    "You analyze food photos for a nutrition diary.",
    "Identify only visible dishes and food products.",
    "Estimate portion weight in grams.",
    "Estimate calories, proteins, fats, and carbs.",
    `Respond strictly in ${responseLanguageName}.`,
    `All user-visible string values must be in ${responseLanguageName}: product names, product descriptions, warnings, empty-result messages, comments, and explanations.`,
    "Keep JSON keys unchanged in English.",
    "Do not translate JSON keys.",
    "Do not mix languages.",
    "Return only JSON matching the provided schema.",
    "Do not add products that are not visible.",
    "If a dish is complex, you may return it as one item, for example a single soup or pasta dish, instead of splitting every ingredient.",
    "Portion weight and macros are estimates.",
    "If the portion is unclear, estimate it, lower confidence, and add a warning.",
    "If food is not recognized, return an empty items array and a warning."
  ].join("\n");
}

function normalizeAnalysis(raw, responseLanguageName) {
  const items = raw.items
    .map((item) => normalizeItem(item, responseLanguageName))
    .filter(Boolean);

  const warnings = raw.warnings
    .map(normalizeText)
    .filter(Boolean);

  if (items.length === 0 && warnings.length === 0) {
    warnings.push(localizedText(responseLanguageName).emptyRecognitionWarning);
  }

  if (items.length > 0 && warnings.length === 0) {
    warnings.push(localizedText(responseLanguageName).defaultWarning);
  }

  return {
    items,
    warnings
  };
}

function normalizeItem(item, responseLanguageName) {
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
    note: normalizeText(item.note) || localizedText(responseLanguageName).defaultNote
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

function resolveRequestedLanguageCode(body) {
  const requested = normalizeText(body?.languageCode) || normalizeText(body?.locale) || "en";
  return requested
    .toLowerCase()
    .split(/[-_]/)[0]
    .trim() || "en";
}

function resolveResponseLanguageName(requestedLanguageCode) {
  return requestedLanguageCode === "ru" ? RESPONSE_LANGUAGE_RUSSIAN : RESPONSE_LANGUAGE_ENGLISH;
}

function localizedText(responseLanguageName) {
  return LOCALIZED_TEXT[responseLanguageName] || LOCALIZED_TEXT[RESPONSE_LANGUAGE_ENGLISH];
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
