// Enable this to get cpu stats
#define COLLECT_CPU_STATS 1

#ifdef __cplusplus
extern "C" {
#endif
void warm_up(void *ptr);
extern int run_inference(void *ptr);
#ifdef __cplusplus
}
#endif
