#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

void SetMaterialSortEnabled(bool v);
bool GetMaterialSortEnabled();

void SetStreamCompactionEnabled(bool v);
bool GetStreamCompactionEnabled();

void SetBVHEnabled(bool v);
bool GetBVHEnabled();

void SetRREnabled(bool v);
bool GetRREnabled();
void SetRRMinDepth(int d);
int  GetRRMinDepth();