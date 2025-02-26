DROP TABLE IF EXISTS vehicle;
CREATE TABLE "vehicle"(
    "id" SERIAL NOT NULL,
    "plate" VARCHAR(255) NOT NULL,
    "model" VARCHAR(255) NOT NULL,
    "color" VARCHAR(255) NOT NULL,
    "brand" VARCHAR(255) NOT NULL,
    "year" BIGINT NOT NULL,
    "owner" VARCHAR(255) NOT NULL,
    "created_at" DATE NOT NULL,
    "updated_at" DATE NOT NULL,
    "deleted_at" DATE NULL
);

DROP TABLE IF EXISTS vehicle_log;
CREATE TABLE "vehicle_log"(
    "id" SERIAL NOT NULL,
    "vehicle_id" BIGINT NOT NULL,
    "severity" VARCHAR(255) NOT NULL,
    "type" VARCHAR(255) NOT NULL,
    "description" VARCHAR(255) NOT NULL,
    "created_at" DATE NOT NULL,
    "updated_at" DATE NOT NULL,
    "deleted_at" DATE NULL
);