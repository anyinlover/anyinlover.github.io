---
title: Instruction Set Architecture
description: Learn about the instruction set architecture, its features, and design considerations.
sidebar_position: 2
---

The objectives of this module is to understand the importance of the instruction set architecture, discuss the features that need to be considered when designing the instruction set architecture of a machine and look at an example ISA, MIPS.

We’ve already seen that the computer architecture course consists of two components – the instruction set architecture and the computer organization itself. The ISA specifies what the processor is capable of doing and the ISA, how it gets accomplished. So the instruction set architecture is basically the interface between your hardware and the software. The only way that you can interact with the hardware is the instruction set of the processor. To command the computer, you need to speak its language and the instructions are the words of a computer’s language and the instruction set is basically its vocabulary. Unless you know the vocabulary and you have a very good vocabulary, you cannot gain good benefits out of the machine. ISA is the portion of the machine which is visible to either the assembly language programmer or a compiler writer or an application programmer. It is the only interface that you have, because the instruction set architecture is the specification of what the computer can do and the machine has to be fabricated in such a way that it will execute whatever has been specified in your ISA. The only way that you can talk to your machine is through the ISA. This gives you an idea of the interface between the hardware and software.

Let us assume you have a high-level program written in C which is independent of the architecture on which you want to work. This high-level program has to be translated into an assembly language program which is specific to a particular architecture. Let us say you find that this consists of a number of instructions like LOAD, STORE, ADD, etc., where, whatever you had written in terms of high-level language now have been translated into a set of instructions which are specific to the specific architecture. All these instructions that are being shown here are part of the instruction set architecture of the MIPS architecture. These are all English like and this is not understandable to the processor because the processor is after all made up of digital components which can understand only zeros and ones. So this assembly language will have to be finely translated into machine language, object code which consists of zeros and ones. So the translation from your high-level language to your assembly language and the binary code will have to be done with the compiler and the assembler.

We shall look at the instruction set features, and see what will go into the zeros and ones and how to interpret the zeros and ones, as data, or instructions or address. The ISA that is designed should last through many implementations, it should have portability, it should have compatibility, it should be used in many different ways so it should have generality and it should also provide convenient functionality to other levels. The taxonomy of ISA is given below.

**Taxonomy**

ISAs differ based on the internal storage in a processor. Accordingly, the ISA can be classified as follows, based on where the operands are stored and whether they are named explicitly or implicitly:

-   **Single accumulator organization**, which names one of the general purpose registers as the accumulator and uses it to necessarily store one of the operands. This indicates that one of the operands is implied to be in the accumulator and it is enough if the other operand is specified along with the instruction.
-   **General register organization,** which specifies all the operands explicitly. Depending on whether the operands are available in memory or registers, it can be further classified as

– **Register** **–** **register**, where registers are used for storing operands. Such architectures are in fact also called **load** **–** **store** architectures, as only load and store instructions can have memory operands.

           **–  Register** **–** **memory**, where one operand is in a register and the other one in memory.

            **– Memory** **–** **memory**, where all the operands are specified as memory operands.

-   **Stack organization**, where the operands are put into the stack and the operations are carried out on the top of the stack. The operands are implicitly specified here.

Let us assume you have to perform the operation A = B + C, where all three operands are memory operands. In the case of an accumulator-based ISA, where we assume that one of the general-purpose registers is being designated as an accumulator and one of the operands will always be available in the accumulator, you have to initially load one operand into the accumulator and the ADD instruction will only specify the operand’s address. In the GPR based ISA, you have three different classifications. In the register memory ISA, One operand has to be moved into any register and the other one can be a memory operand. In the register – register ISA, both operands will have to moved to two registers and the ADD instruction will only work on registers. The memory – memory ISA permits both memory operands. So you can directly add. In a stack-based ISA, you’ll have to first of all push both operands onto the stack and then simply give an add instruction which will add the top two elements of the stack and then store the result in the stack. So you can see from these examples that you have different ways of executing the same operation, and it obviously depends upon the ISA. Among all these ISAs, It is the register – register ISA that is very popular and used in all RISC architectures.

We shall now look at what are the different features that need to be considered when designing the instruction set architecture. They are:

-   Types of instructions (Operations in the Instruction set)
-   Types and sizes of operands
-   Addressing Modes
-   Addressing Memory
-   Encoding and Instruction Formats
-   Compiler related issues

First of all, you have to decide on the types of instructions, i.e. what are the various instructions that you want to support in the ISA. The tasks carried out by a computer program consist of a sequence of small steps, such as multiplying two numbers, moving a data from a register to a memory location, testing for a particular condition like zero, reading a character from the input device or sending a character to be displayed to the output device, etc.. A computer must have the following types of instructions:

-   Data transfer instructions
-   Data manipulation instructions
-   Program sequencing and control instructions
-   Input and output instructions

Data transfer instructions perform data transfer between the various storage places in the computer system, viz. registers, memory and I/O. Since, both the instructions as well as data are stored in memory, the processor needs to read the instructions and data from memory. After processing, the results must be stored in memory. Therefore, two basic operations involving the memory are needed, namely, *Load* (or *Read* or *Fetch*) and *Store* (or *Write*). The Load operation transfers a copy of the data from the memory to the processor and the Store operation moves the data from the processor to memory. Other data transfer instructions are needed to transfer data from one register to another or from/to I/O devices and the processor.

Data manipulation instructions perform operations on data and indicate the computational capabilities for the processor. These operations can be arithmetic operations, logical operations or shift operations. Arithmetic operations include addition (with and without carry), subtraction (with and without borrow), multiplication, division, increment, decrement and finding the complement of a number. The logical and bit manipulation instructions include AND, OR, XOR, Clear carry, set carry, etc. Similarly, you can perform different types of shift and rotate operations.

We generally assume a sequential flow of instructions. That is, instructions that are stored in consequent locations are executed one after the other. However, you have program sequencing and control instructions that help you change the flow of the program. This is best explained with an example. Consider the task of adding a list of *n* numbers. A possible sequence is given below.

Move DATA1, R0

Add DATA2, R0

Add DATA3, R0

Add DATAn, R0

Move R0, SUM

The addresses of the memory locations containing the *n* numbers are symbolically given as DATA1, DATA2, . . , DATAn, and a separate Add instruction is used to add each Databer to the contents of register R0. After all the numbers have been added, the result is placed in memory location SUM. Instead of using a long list of Add instructions, it is possible to place a single Add instruction in a program loop, as shown below:

Move N, R1

Clear R0

LOOP Determine address of “Next” number and add “Next” number to R0

Decrement R1

Branch > 0, LOOP

Move R0, SUM

The loop is a straight-line sequence of instructions executed as many times as needed. It starts at location LOOP and ends at the instruction Branch*\>*0\. During each pass through this loop, the address of the next list entry is determined, and that entry is fetched and added to R0. The address of an operand can be specified in various ways, as will be described in the next section. For now, you need to know how to create and control a program loop. Assume that the number of entries in the list, n, is stored in memory location N. Register R1 is used as a counter to determine the number of times the loop is executed. Hence, the contents of location N are loaded into register R1 at the beginning of the program. Then, within the body of the loop, the instruction, Decrement R1 reduces the contents of R1 by 1 each time through the loop. The execution of the loop is repeated as long as the result of the decrement operation is greater than zero.

You should now be able to understand *branch* instructions. This type of instruction loads a new value into the program counter. As a result, the processor fetches and executes the instruction at this new address, called the *branch target,* instead of the instruction at the location that follows the branch instruction in sequential address order. The branch instruction can be conditional or unconditional. An *unconditional branch* instruction does a branch to the specified address irrespective of any condition. A *conditional branch* instruction causes a branch only if a specified condition is satisfied. If the condition is not satisfied, the PC is incremented in the normal way, and the next instruction in sequential address order is fetched and executed. In the example above, the instruction Branch*\>*0 LOOP (branch if greater than 0) is a conditional branch instruction that causes a branch to location LOOP if the result of the immediately preceding instruction, which is the decremented value in register R1, is greater than zero. This means that the loop is repeated as long as there are entries in the list that are yet to be added to R0. At the end of the *n*th pass through the loop, the Decrement instruction produces a value of zero, and, hence, branching does not occur. Instead, the Move instruction is fetched and executed. It moves the final result from R0 into memory location SUM. Some ISAs refer to such instructions as *Jumps*. The processor keeps track of information about the results of various operations for use by subsequent conditional branch instructions. This is accomplished by recording the required information in individual bits, often called *condition code flags*. These flags are usually grouped together in a special processor register called the *condition code register* or *status register*. Individual condition code flags are set to 1 or cleared to 0, depending on the outcome of the operation performed. Some of the commonly used flags are: Sign, Zero, Overflow and Carry. The call and return instructions are used in conjunction with subroutines. A subroutine is a self-contained sequence of instructions that performs a given computational task. During the execution of a program, a subroutine may be called to perform its function many times at various points in the main program. Each time a subroutine is called, a branch is executed to the beginning of the subroutine to start executing its set of instructions. After the subroutine has been executed, a branch is made back to the main program, through the return instruction. Interrupts can also change the flow of a program. A program interrupt refers to the transfer of program control from a currently running program to another service program as a result of an external or internally generated request. Control returns to the original program after the service program is executed. The interrupt procedure is, in principle, quite similar to a subroutine call except for three variations: (1) The interrupt is usually initiated by an internal or external signal apart from the execution of an instruction (2) the address of the interrupt service program is determined by the hardware or from some information from the interrupt signal or the instruction causing the interrupt; and (3) an interrupt procedure usually stores all the information necessary to define the state of the CPU rather than storing only the program counter. Therefore, when the processor is interrupted, it saves the current status of the processor, including the return address, the register contents and the status information called the Processor Status Word (PSW), and then jumps to the interrupt handler or the interrupt service routine. Upon completing this, it returns to the main program. Interrupts are handled in detail in the next unit on Input / Output.

Input and Output instructions are used for transferring information between the registers, memory and the input / output devices. It is possible to use special instructions that exclusively perform I/O transfers, or use memory – related instructions itself to do I/O transfers.

Suppose you are designing an embedded processor which is meant to be performing a particular application, then definitely you will have to bring instructions which are specific to that particular application. When you’re designing a general-purpose processor, you only look at including all general types of instructions. Examples of specialized instructions may be media and signal processing related instructions, say vector type of instructions which try to exploit the data level parallelism, where the same operation of addition or subtraction is going to be done on different data and then you may have to look at saturating arithmetic operations, multiply and accumulator instructions.

The data types and sizes indicate the various data types supported by the processor and their lengths. Common operand types – Character (8 bits), Half word (16 bits), Word (32 bits), Single Precision Floating Point (1 Word), Double Precision Floating Point (2 Words), Integers – two’s complement binary numbers, Characters usually in ASCII, Floating point numbers following the IEEE Standard 754 and Packed and unpacked decimal numbers.

**Addressing Modes**

The operation field of an instruction specifies the operation to be performed. This operation must be executed on some data that is given straight away or stored in computer registers or memory words. The way the operands are chosen during program execution is dependent on the *addressing mode* of the instruction. The addressing mode specifies a rule for interpreting or modifying the address field of the instruction before the operand is actually referenced. In this section, you will learn the most important addressing modes found in modern processors.

Computers use addressing mode techniques for the purpose of accommodating one or both of the following:

1\. To give programming versatility to the user by providing such facilities as pointers to memory, counters for loop control, indexing of data, and program relocation.

2\. To reduce the number of bits in the addressing field of the instruction.

When you write programs in a high-level language, you use constants, local and global variables, pointers, and arrays. When translating a high-level language program into assembly language, the compiler must be able to implement these constructs using the facilities provided in the instruction set of the computer in which the program will be run. The different ways in which the location of an operand is specified in an instruction are referred to as addressing modes. Variables and constants are the simplest data types and are found in almost every computer program. In assembly language, a variable is represented by allocating a register or a memory location to hold its value.

**Register mode** *—* The operand is the contents of a processor register; the name (address) of the register is given in the instruction.

**Absolute mode** *—* The operand is in a memory location; the address of this location is given explicitly in the instruction. This is also called **Direct**.

Address and data constants can be represented in assembly language using the Immediate mode.

**Immediate mode** *—* The operand is given explicitly in the instruction. For example, the instruction Move 200*immediate,* R0 places the value 200 in register R0. Clearly, the Immediate mode is only used to specify the value of a source operand. A common convention is to use the sharp sign (#) in front of the value to indicate that this value is to be used as an immediate operand. Hence, we write the instruction above in the form Move #200, R0. Constant values are used frequently in high-level language programs. For example, the statement A = B + 6 contains the constant 6. Assuming that A and B have been declared earlier as variables and may be accessed using the Absolute mode, this statement may be compiled as follows:

    Move B, R1

Add #6, R1

Move R1, A

Constants are also used in assembly language to increment a counter, test for some bit pattern, and so on.

**Indirect mode** *—* In the addressing modes that follow, the instruction does not give the operand or its address explicitly. Instead, it provides information from which the memory address of the operand can be determined. We refer to this address as the *effective* *address* (EA) of the operand. In this mode, the effective address of the operand is the contents of a register or memory location whose address appears in the instruction. We denote indirection by placing the name of the register or the memory address given in the instruction in parentheses. For example, consider the instruction, Add (R1), R0. To execute the Add instruction, the processor uses the value in register R1 as the effective address of the operand. It requests a read operation from the memory to read the contents of this location. The value read is the desired operand, which the processor adds to the contents of register R0. Indirect addressing through a memory location is also possible as indicated in the instruction Add (A), R0. In this case, the processor first reads the contents of memory location A, then requests a second read operation using this value as an address to obtain the operand. The register or memory location that contains the address of an operand is called a *pointer*. Indirection and the use of pointers are important and powerful concepts in programming. Changing the contents of location A in the example fetches different operands to add to register R0.

**Index mode** *—* The next addressing mode you learn provides a different kind of flexibility for accessing operands. It is useful in dealing with lists and arrays. In this mode, the effective address of the operand is generated by adding a constant value (displacement) to the contents of a register. The register used may be either a special register provided for this purpose, or may be any one of the general-purpose registers in the processor. In either case, it is referred to as an *index register*. We indicate the Index mode symbolically as X(Ri ), where X denotes the constant value contained in the instruction and Ri is the name of the register involved. The effective address of the operand is given by EA = X + \[Ri\]. The contents of the index register are not changed in the process of generating the effective address. In an assembly language program, the constant X may be given either as an explicit number or as a symbolic name representing a numerical value. When the instruction is translated into machine code, the constant X is given as a part of the instruction and is usually represented by fewer bits than the word length of the computer. Since X is a signed integer, it must be sign-extended to the register length before being added to the contents of the register.

**Relative mode** *—* The above discussion defined the Index mode using general-purpose processor registers. A useful version of this mode is obtained if the program counter, PC, is used instead of a general purpose register. Then, X (PC) can be used to address a memory location that is X bytes away from the location presently pointed to by the program counter. Since the addressed location is identified “relative” to the program counter, which always identifies the current execution point in a program, the name Relative mode is associated with this type of addressing. In this case, the effective address is determined by the Index mode using the program counter in place of the general-purpose register Ri. This addressing mode is generally used with control flow instructions.

Though this mode can be used to access data operands. But, its most common use is to specify the target address in branch instructions. An instruction such as Branch *\>* 0 LOOP, which we discussed earlier, causes program execution to go to the branch target location identified by the name LOOP if the branch condition is satisfied. This location can be computed by specifying it as an offset from the current value of the program counter. Since the branch target may be either before or after the branch instruction, the offset is given as a signed number. Recall that during the execution of an instruction, the processor increments the PC to point to the next instruction. Most computers use this updated value in computing the effective address in the Relative mode.

The two modes described next are useful for accessing data items in successive locations in the memory.

**Autoincrement mode** *—* The effective address of the operand is the contents of a register specified in the instruction. After accessing the operand, the contents of this register are automatically incremented to point to the next item in a list. We denote the Autoincrement mode by putting the specified register in parentheses, to show that the contents of the register are used as the effective address, followed by a plus sign to indicate that these contents are to be incremented after the operand is accessed. Thus, the Autoincrement mode is written as **(Ri )+.**

**Autodecrement mode** *—* As a companion for the Autoincrement mode, another useful mode accesses the items of a list in the reverse order. In the autodecrement mode, the contents of a register specified in the instruction are first automatically decremented and are then used as the effective address of the operand. We denote the Autodecrement mode by putting the specified register in parentheses, preceded by a minus sign to indicate that the contents of the register are to be decremented before being used as the effective address. Thus, we write **– (Ri ).** In this mode, operands are accessed in descending address order. You may wonder why the address is decremented before it is used in the Autodecrement mode and incremented after it is used in the Autoincrement mode. The main reason for this is that these two modes can be used together to implement a stack.

**Instruction Formats**

The previous sections have shown you that the processor can execute different types of instructions and there are different ways of specifying the operands. Once all this is decided, this information has to be presented to the processor in the form of an instruction format. The number of bits in the instruction is divided into groups called fields. The most common fields found in instruction formats are

1\. An operation code field that specifies the operation to be performed. The number of bits will indicate the number of operations that can be performed.

2\. An address field that designates a memory address or a processor register. The number of bits depends on the size of memory or the number of registers.

3\. A mode field that specifies the way the operand or the effective address is determined. This depends on the number of addressing modes supported by the processor.

The number of address fields may be three, two or one depending on the type of ISA used. Also, observe that, based on the number of operands that are supported and the size of the various fields, the length of the instructions will vary. Some processors fit all the instructions into a single sized format, whereas others make use of formats of varying sizes. Accordingly, you have a fixed format or a variable format.

Interpreting memory addresses – you basically have two types of interpretation of the memory addresses – Big endian arrangement and the little endian arrangement. Memories are normally arranged as bytes and a unique address of a memory location is capable of storing 8 bits of information. But when you look at the word length of the processor, the word length of the processor may be more than one byte. Suppose you look at a 32-bit processor, it is made up of four bytes. These four bytes span over four memory locations. When you specify the address of a word how you would specify the address of the word – are you going to specify the address of the most significant byte as the address of the word (big end) or specify the address of the least significant byte (little end) as the address of the word. That distinguishes between a big endian arrangement and a little endian arrangement. IBM, Motorola, HP follow the big endian arrangement and Intel follows the little endian arrangement. Also, when a data spans over different memory locations, and if you try to access a word which is aligned with the word boundary, we say there is an alignment. If you try to access the words not starting at a word boundary, you can still access, but they are not aligned. Whether there is support to access data that is misaligned is a design issue. Even if you’re allowed to access data that is misaligned, it normally takes more number of memory cycles to access the data.

Finally looking at the role of compilers the compiler has a lot of role to play when you’re defining the instruction set architecture. Gone are the days where people thought that compilers and architectures are going to be independent of each other. Only when the compiler knows the internal architecture of the processor it’ll be able to produce optimised code. So the architecture will have to expose itself to the compiler and the compiler will have to make use of whatever hardware is exposed. The ISA should be compiler friendly. The basic ways in which the ISA can help the compiler are regularity, orthogonality and the ability to weigh different options.

Finally, all the features of an ISA are discussed with respect to the 80×86 and MIPS.

1. *Class of ISA* — Nearly all ISAs today are classified as general-purpose register architectures, where the operands are either registers or memory locations. The 80×86 has 16 general-purpose registers and 16 that can hold floating point data, while MIPS has 32 general-purpose and 32 floating-point registers. The two popular versions of this class are *register-memory* ISAs such as the 80×86, which can access memory as part of many instructions, and *load-store* ISAs such as MIPS, which can access memory only with load or store instructions. All recent ISAs are load-store.

2. *Memory addressing* — Virtually all desktop and server computers, including the 80×86 and MIPS, use byte addressing to access memory operands. Some architectures, like MIPS, require that objects must be *aligned*. An access to an object of size *s* bytes at byte address *A* is aligned if *A* mod *s =* 0. The 80×86 does not require alignment, but accesses are generally faster if operands are aligned.

3. *Addressing modes* — In addition to specifying registers and constant operands, addressing modes specify the address of a memory object. MIPS addressing modes are Register, Immediate (for constants), and Displacement, where a constant offset is added to a register to form the memory address. The 80×86 supports those three plus three variations of displacement: no register (absolute), two registers (based indexed with displacement), two registers where one register is multiplied by the size of the operand in bytes (based with scaled index and displacement). It has more like the last three, minus the displacement field: register indirect, indexed, and based with scaled index.

4. *Types and sizes of operands* — Like most ISAs, MIPS and 80×86 support operand sizes of 8-bit (ASCII character), 16-bit (Unicode character or half word), 32-bit (integer or word), 64-bit (double word or long integer), and IEEE 754 floating point in 32-bit (single precision) and 64-bit (double precision). The 80×86 also supports 80-bit floating point (extended double precision).

5. *Operations* — The general categories of operations are data transfer, arithmetic logical, control, and floating point. MIPS is a simple and easy-to-pipeline instruction set architecture, and it is representative of the RISC architectures being used in 2006. The 80×86 has a much richer and larger set of operations.

6. *Control flow instructions* — Virtually all ISAs, including 80×86 and MIPS, support conditional branches, unconditional jumps, procedure calls, and returns. Both use PC-relative addressing, where the branch address is specified by an address field that is added to the PC. There are some small differences.MIPS conditional branches (BE, BNE, etc.) test the contents of registers, while the 80×86 branches (JE, JNE, etc.) test condition code bits set as side effects of arithmetic/logic operations. MIPS procedure call (JAL) places the return address in a register, while the 80×86 call (CALLF) places the return address on a stack in memory.

7.  *Encoding an ISA* — There are two basic choices on encoding: *fixed length* and v*ariable length*. All MIPS instructions are 32 bits long, which simplifies instruction decoding (shown below). The 80×86 encoding is variable length, ranging from 1 to 18 bytes. Variable length instructions can take less space than fixed-length instructions, so a program compiled for the 80×86 is usually smaller than the same program compiled for MIPS. Note that choices mentioned above will affect how the instructions are encoded into a binary representation. For example, the number of registers and the number of addressing modes both have a significant impact on the size of instructions, as the register field and addressing mode field can appear many times in a single instruction.

To summarize, we have looked at the taxonomy of ISAs and the various features that need to be decided while designing the ISA. We also looked at example ISAs, the MIPS ISA and the 80×86 ISA.

**Web Links / Supporting Materials**

-   [http://en.wikipedia.org/wiki/Instruction\_set](http://en.wikipedia.org/wiki/Instruction_set)
-   Computer Architecture – A Quantitative Approach , John L. Hennessy and David A. Patterson, 5th.Edition, Morgan Kaufmann, Elsevier, 2011.
-   Computer Organization and Design – The Hardware / Software Interface, David A. Patterson and John L. Hennessy, 4th.Edition, Morgan Kaufmann, Elsevier, 2009.
-   Computer Organization, Carl Hamacher, Zvonko Vranesic and Safwat Zaky, 5th.Edition, McGraw-Hill Higher Education, 2011.
