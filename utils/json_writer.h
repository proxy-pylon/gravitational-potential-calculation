/*
 * json_writer.h - Simple JSON writer for CUDA C programs
 *
 * Provides functions to write structured JSON output for experiment results
 */

#ifndef JSON_WRITER_H
#define JSON_WRITER_H

#include <stdio.h>
#include <time.h>

/*
 * Get current timestamp in ISO 8601 format
 */
void get_timestamp(char *buffer, size_t buffer_size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(buffer, buffer_size, "%Y-%m-%dT%H:%M:%S", t);
}

/*
 * Write JSON results for direct summation method
 */
void write_json_direct(
    const char *filename,
    const char *implementation,  // "cuda" or "cupy"
    int n_bodies,
    float G,
    float softening,
    float mean_time_ms,
    float std_time_ms,
    float min_time_ms,
    float max_time_ms,
    int n_trials,
    float U_computed,
    float U_analytical,
    float relative_error,
    float mean_pot,
    float std_pot,
    float min_pot,
    float max_pot,
    const char *input_file,
    const char *output_file)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Warning: Could not write JSON file: %s\n", filename);
        return;
    }

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"method\": \"direct_summation\",\n");
    fprintf(fp, "  \"implementation\": \"%s\",\n", implementation);
    fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp);

    fprintf(fp, "  \"parameters\": {\n");
    fprintf(fp, "    \"n_bodies\": %d,\n", n_bodies);
    fprintf(fp, "    \"G\": %.6f,\n", G);
    fprintf(fp, "    \"softening\": %.6f\n", softening);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"timing\": {\n");
    fprintf(fp, "    \"mean_time_seconds\": %.6f,\n", mean_time_ms / 1000.0f);
    fprintf(fp, "    \"std_time_seconds\": %.6f,\n", std_time_ms / 1000.0f);
    fprintf(fp, "    \"min_time_seconds\": %.6f,\n", min_time_ms / 1000.0f);
    fprintf(fp, "    \"max_time_seconds\": %.6f,\n", max_time_ms / 1000.0f);
    fprintf(fp, "    \"n_trials\": %d,\n", n_trials);
    fprintf(fp, "    \"warmup_runs\": 3\n");
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"results\": {\n");
    fprintf(fp, "    \"U_computed\": %.6f,\n", U_computed);
    fprintf(fp, "    \"U_analytical\": %.6f,\n", U_analytical);
    fprintf(fp, "    \"relative_error\": %.6e,\n", relative_error);
    fprintf(fp, "    \"relative_error_percent\": %.4f\n", relative_error * 100.0f);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"potential_statistics\": {\n");
    fprintf(fp, "    \"mean\": %.6f,\n", mean_pot);
    fprintf(fp, "    \"std\": %.6f,\n", std_pot);
    fprintf(fp, "    \"min\": %.6f,\n", min_pot);
    fprintf(fp, "    \"max\": %.6f\n", max_pot);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"files\": {\n");
    fprintf(fp, "    \"input\": \"%s\",\n", input_file);
    fprintf(fp, "    \"output\": \"%s\"\n", output_file);
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");

    fclose(fp);
}

/*
 * Write JSON results for Monte Carlo method
 */
void write_json_mc(
    const char *filename,
    const char *implementation,  // "cuda" or "cupy"
    int n_bodies,
    int n_samples,
    float G,
    float softening,
    int seed,
    float mean_time_ms,
    float std_time_ms,
    float min_time_ms,
    float max_time_ms,
    int n_trials,
    float U_computed,
    float U_analytical,
    float relative_error,
    float mean_pot,
    float std_pot,
    float min_pot,
    float max_pot,
    const char *input_file,
    const char *output_file)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Warning: Could not write JSON file: %s\n", filename);
        return;
    }

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"method\": \"monte_carlo\",\n");
    fprintf(fp, "  \"implementation\": \"%s\",\n", implementation);
    fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp);

    fprintf(fp, "  \"parameters\": {\n");
    fprintf(fp, "    \"n_bodies\": %d,\n", n_bodies);
    fprintf(fp, "    \"n_samples\": %d,\n", n_samples);
    fprintf(fp, "    \"G\": %.6f,\n", G);
    fprintf(fp, "    \"softening\": %.6f,\n", softening);
    fprintf(fp, "    \"seed\": %d\n", seed);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"timing\": {\n");
    fprintf(fp, "    \"mean_time_seconds\": %.6f,\n", mean_time_ms / 1000.0f);
    fprintf(fp, "    \"std_time_seconds\": %.6f,\n", std_time_ms / 1000.0f);
    fprintf(fp, "    \"min_time_seconds\": %.6f,\n", min_time_ms / 1000.0f);
    fprintf(fp, "    \"max_time_seconds\": %.6f,\n", max_time_ms / 1000.0f);
    fprintf(fp, "    \"n_trials\": %d,\n", n_trials);
    fprintf(fp, "    \"warmup_runs\": 3\n");
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"results\": {\n");
    fprintf(fp, "    \"U_computed\": %.6f,\n", U_computed);
    fprintf(fp, "    \"U_analytical\": %.6f,\n", U_analytical);
    fprintf(fp, "    \"relative_error\": %.6e,\n", relative_error);
    fprintf(fp, "    \"relative_error_percent\": %.4f\n", relative_error * 100.0f);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"potential_statistics\": {\n");
    fprintf(fp, "    \"mean\": %.6f,\n", mean_pot);
    fprintf(fp, "    \"std\": %.6f,\n", std_pot);
    fprintf(fp, "    \"min\": %.6f,\n", min_pot);
    fprintf(fp, "    \"max\": %.6f\n", max_pot);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"files\": {\n");
    fprintf(fp, "    \"input\": \"%s\",\n", input_file);
    fprintf(fp, "    \"output\": \"%s\"\n", output_file);
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");

    fclose(fp);
}

#endif // JSON_WRITER_H
