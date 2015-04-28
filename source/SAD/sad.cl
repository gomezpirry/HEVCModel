#define BLOCK_WIDTH 4

__kernel void calcSAD(  __global short* const block_pixel,    __global short* const area_pixel, __local uint* sadHorizontal, __local uint* sadVertical, __local uint* sadAMP, const short CtuSize, const short stride,     
/*         MxM                        MxM/2                        M/2xM                      MxM/4(Up)                  MxM/4(Down)                   Mx3M/4(Up)              Mx3M/4(Down)                      M/4xM(Left)                 M/4xM(Right)                 3M/4xM(Left)                   3M/4xM                                                                                    */  
  __global uint* sad8x8,      __global uint* sad8x4,      __global uint* sad4x8,              
  __global uint* sad16x16,    __global uint* sad16x8,     __global uint* sad8x16,     __global uint* sad16x4U,    __global uint* sad16x4D,    __global uint* sad16x12U,   __global uint* sad16x12D,      __global uint* sad4x16L,     __global uint* sad4x16R,    __global uint* sad12x16L,   __global uint* sad12x16R, 
  __global uint* sad32x32,    __global uint* sad32x16,    __global uint* sad16x32,    __global uint* sad32x8U,    __global uint* sad32x8D,    __global uint* sad32x24U,   __global uint* sad32x24D,      __global uint* sad8x32L,     __global uint* sad8x32R,    __global uint* sad24x32L,   __global uint* sad24x32R,
  __global uint* sad64x64,    __global uint* sad64x32,    __global uint* sad32x64,    __global uint* sad64x16U,   __global uint* sad64x16D,   __global uint* sad64x48U,   __global uint* sad64x48D,      __global uint* sad16x64L,    __global uint* sad16x64R,   __global uint* sad48x64L,   __global uint* sad48x64R)

{   
    unsigned int globalId = get_global_id(0);

    /* get indices relative to the work-item i 16x16 matrix */
    unsigned int i  = get_local_id(0);
    unsigned int j  = get_local_id(1); 
    
    /* get size of each row of work items */
    unsigned int group = get_local_size(0);

    /* index to convert pos (x,y) of each work-item in pos (i) */
    int sum = 0;

    const int index = i+(group*j);

    /* index to put index in each 4x4 block in CTU block */
    uint idx = j * (CtuSize*BLOCK_WIDTH) + i * 4;

    /* cycle that calculate the sad for each 4x4 block */
    for(int x = 0; x<BLOCK_WIDTH; x++)
    { 
        const int idy =  idx + (x * CtuSize); //runs the block in x
        sum += abs_diff(block_pixel[idy],area_pixel[idy]); 
        sum += abs_diff(block_pixel[idy + 1],area_pixel[idy + 1]);
        sum += abs_diff(block_pixel[idy + 2],area_pixel[idy + 2]);
        sum += abs_diff(block_pixel[idy + 3],area_pixel[idy + 3]);// calculate sad por each pixel and runs the block in y
    } 
    barrier(CLK_GLOBAL_MEM_FENCE); 
    
    
    /* allocate the result the each 4x4 sad in local memory for calculate vertical recursive sad*/
    sadVertical[index] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
        
    /*create index to move in horizontal and vertical positions*/
    uint indexH = index * 2;
    uint indexV = (j * 32 + i);
    uint idxV = (j*4) + i;
    
    /* calculate sad of 128 blocks of 8x4 and transfer to 8x4 buffer*/
    sadHorizontal[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    sad8x4[index] = sadHorizontal[index];
    
    /* calculate sad of 128 blocks of 4x8 and transfer to 4x8 buffer*/
    sadVertical[index] = sadVertical[indexV] + sadVertical[indexV+16];
    
    sad4x8[index] = sadVertical[index];
    
    /*calculate all 16x4 blocks*/
    sadAMP[index] = sadHorizontal[indexH] + sadHorizontal[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*calculate 16x4(Up), 16x4(Down), 16x12(Up) and 16x12(Down) blocks*/
    indexV = (j * 16 + i);
    
    if(i<4)
    {
        //16x4(Up)
        sad16x4U[idxV] = sadAMP[indexV];
        //16x4(Down)
        sad16x4D[idxV] = sadAMP[indexV + 12];
        //16x12(Up)
        sad16x12U[idxV] = sadAMP[indexV] + sadAMP[indexV + 4] + sadAMP[indexV + 8];
        //16x12(Down)
        sad16x12D[idxV] = sadAMP[indexV + 4] + sadAMP[indexV + 8] + sadAMP[indexV +12];
    }
    
    /*calculate all 4x16 blocks*/
    indexV = (j * 32) + i;
    sadAMP[index] = sadVertical[indexV] + sadVertical[indexV + 16];  
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*calculate 4x16(Left), 4x16Down), 16x12(Up) and 16x12(Down) blocks*/
    
    if(i<4)
    {
        //4x16(Left)
        sad4x16L[idxV] = sadAMP[idxV * 4];
        //4x16(Rigth)
        sad4x16R[idxV] = sadAMP[(idxV * 4) + 3];
        //12x16(Left)
        sad12x16L[idxV] = sadAMP[(idxV * 4)] + sadAMP[(idxV * 4) + 1] + sadAMP[(idxV * 4) + 2];
        //12x16(Rigth)
        sad12x16R[idxV] = sadAMP[(idxV *4) + 1] + sadAMP[(idxV * 4) + 2] + sadAMP[(idxV * 4) + 3];        
    }
    
    /* calculate sad of 64 blocks of 8x8 and transfer to 8x8 buffer */
    sadVertical[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    sad8x8[index] = sadVertical[index];
    
    
    /* calculate sad of 32 blocks of 16x8 and transfer to 16x8 buffer)*/ 
    sadHorizontal[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
       
    sad16x8[index] = sadHorizontal[index];
    
     /* calculate sad of 32 blocks of 8x16 and transfer to 8x16 buffer*/
    indexV = (j * 16 + i);
    idxV = (j * 8) + i;
    if(i<8)
    {
        sadVertical[idxV] = sadVertical[indexV] + sadVertical[indexV + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
     
    sad8x16[index] = sadVertical[index];
    
    /* calculate all 32x8 blocks */
    sadAMP[index] = sadHorizontal[indexH] + sadHorizontal[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    idxV = (j * 2) + i;
    indexV = (j * 8) + i;
    
    /* calculate 32x8(Up), 32x8(Down), 32x24(Up) and 32x24(Down) */
    if(i<2)
    {
        //sad32x8(Up)
        sad32x8U[idxV] = sadAMP[indexV];
        //32x8(Down)
        sad32x8D[idxV] = sadAMP[indexV + 6];        
        //32x24(Up) 
        sad32x24U[idxV] = sadAMP[indexV] + sadAMP[indexV + 2] + sadAMP[indexV + 4]; 
        //32x24(Down)
        sad32x24D[idxV] = sadAMP[indexV + 2] + sadAMP[indexV + 4] + sadAMP[indexV + 6]; 
    }    
    
    /* calculate all 8x32 blocks */
    indexV = (j * 16) + i;
    idxV = (j * 8) + i;
    if(i<8)
    {
        sadAMP[idxV] = sadVertical[indexV] + sadVertical[indexV + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    /* calculate 8x32(Left), 8x32(Rigth), 24x32(Left) and 24x32(Rigth) sad */
    idxV = (j * 2) + i;
    if(i<2)
    {
        //8x32(Left)
         sad8x32L[idxV] = sadAMP[idxV * 4];
        //8x32(Rigth)
         sad8x32R[idxV] = sadAMP[(idxV * 4) + 3];
        //24x32(Left) 
        sad24x32L[idxV] = sadAMP[idxV * 4] + sadAMP[(idxV * 4) + 1] + sadAMP[(idxV * 4) + 2];
        //24x32(Rigth)
        sad24x32R[idxV] = sadAMP[(idxV * 4) + 1] + sadAMP[(idxV * 4) + 2] + sadAMP[(idxV * 4) + 3];
    }
    
     /* calculate sad of 16 blocks of 16x16 (first allocate in temp sadVertical, and later transfer to 16x16 buffer)*/
    sadVertical[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    sad16x16[index] = sadVertical[index];
    barrier(CLK_LOCAL_MEM_FENCE);       
    
    
    /* calculate sad of 8 blocks of 32x16 (transfer directly to 32x16 buffer)*/
    sadHorizontal[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    sad32x16[index] = sadHorizontal[index];
    
      /* calculate sad of 16 blocks of 16x32 (first allocate in temp sadVertical, and later transfer to 16x32 buffer)*/
    indexV = (j * 8 + i);
    if(i<4)
    {
        sadVertical[(j*4) + i] = sadVertical[indexV] + sadVertical[indexV + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    sad16x32[index] = sadVertical[index];
    
    /*calculate all 64x16 blocks*/
    sadAMP[index] = sadHorizontal[indexH] + sadHorizontal[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(i==0 && j==0)
    {   
        //sad64x16(Up)
        sad64x16U[0] = sadAMP[0];
        //sad64x16(Down)
        sad64x16D[0] = sadAMP[3];
        //sad64x48(Up)
        sad64x48U[0] = sadAMP[0] + sadAMP[1] + sadAMP[2];
        //sad64x16(Down)
        sad64x48D[0] = sadAMP[1] + sadAMP[2] + sadAMP[3];
    }
    
    sadAMP[index] =  sadVertical[index] + sadVertical[index + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    
     if(i==0 && j==0)
    {   
        //sad64x16(Up)
        sad16x64L[0] = sadAMP[0];
        //sad64x16(Down)
        sad16x64R[0] = sadAMP[3];
        //sad64x48(Up)
        sad48x64L[0] = sadAMP[0] + sadAMP[1] + sadAMP[2];
        //sad64x16(Down)
        sad48x64R[0] = sadAMP[1] + sadAMP[2] + sadAMP[3];
    }
    
    /* calculate sad of 4 blocks of 32x32 (first allocate in temp sadVertical, and later transfer to 32x32 buffer)*/
    sadVertical[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    sad32x32[index] = sadVertical[index];
    
     /* calculate sad of 2 blocks of 64x32 (transfer directly to 64x32 buffer)*/
    sadHorizontal[index] = sadVertical[indexH] + sadVertical[indexH + 1];
    barrier(CLK_LOCAL_MEM_FENCE);  
    
    sad64x32[index] = sadHorizontal[index];
    
    /* calculate sad of 2 blocks of 32x64 (first allocate in temp sadVertical, and later transfer to 32x64 buffer)*/
    sadVertical[index] = sadVertical[index] + sadVertical[index + 2];
    barrier(CLK_LOCAL_MEM_FENCE);    
    
    sad32x64[index] = sadVertical[index];  
    
    /* calculate sad of 64x64 block and transfer directly to 64x32 buffer)*/
    sad64x64[index] = sadVertical[index] + sadVertical[index + 1];
}

