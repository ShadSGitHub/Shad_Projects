package core;
//@author Janet Baez

import java.util.*;
import java.awt.Color;

public interface ICodemaker 
{
    public void generateSecreteCode();    
    public void checkAttemptedCode(ArrayList <Color> codebreakerAttempt);
}