//
// Created by leo on 2022/1/12.
//
#include "hrnetpose/hrnetpose_model.h"

int main(int argc , char** argvs)
{
    Hrnetpose_model hrnetpose_model;
//    hrnetpose_model.int8_flag=true;
//    hrnetpose_model.serialize_engine();
    hrnetpose_model.deserialize_engine("../hrnet32_256x192_dynamic_int8.engine");
    hrnetpose_model.run_frame();
}

