__kernel void fill_array(__global float* data, float value, int size) {
    int id = get_global_id(0);
    if (id < size) {
        data[id] = data[id] + value + id + 0.3;  // 赋值: value + 索引
    }
}
